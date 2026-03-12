"""
optics.py
=========
Optical characterisation utilities:

  1. Airy disk PSF — analytic diffraction-limited PSF for a circular
     (possibly obstructed) aperture.  Fully determined by scope geometry
     and wavelength; never needs to be estimated from data.

  2. WCS plate-solve utilities — extract camera geometry from a FITS WCS
     solution (position angle, plate scale, flip state) and compute
     sub-pixel frame shifts from WCS differences.

  3. Instrument PSF composition — combine the known diffraction PSF with
     an optional optical-aberration PSF (H_optics, not yet implemented)
     to produce the full instrument PSF used by the stacker.

Physical model
--------------
The total per-frame PSF is:

    H_total,i = H_diff  *  H_optics  *  H_seeing,i

    H_diff    — analytic Airy disk, computed here from scope parameters
    H_optics  — optical aberrations (placeholder, future work)
    H_seeing,i — atmospheric turbulence, estimated per-frame by
                  FrameCharacterizer from stars in the frame

The stacker only needs H_diff and H_optics (both fixed per instrument
setup).  H_seeing,i is estimated per frame and is NOT computed here.

Dependencies
------------
    numpy
    scipy.special  (j1 — Bessel function of the first kind, order 1)
    astropy.wcs    (WCS parsing from FITS headers)
    astropy.io.fits
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.special import j1
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


# ============================================================================
# ScopeGeometry — scope + camera parameters
# ============================================================================

@dataclass
class ScopeGeometry:
    """
    Physical parameters of the telescope + camera combination.

    All lengths in the units shown; conversions are handled internally.

    Parameters
    ----------
    aperture_mm : float
        Clear aperture diameter in millimetres.
        For the Esprit 100ED: 100.0

    focal_length_mm : float
        Effective focal length in millimetres.
        For the Esprit 100ED: 550.0
        Use the WCS-measured plate scale to refine this if a flattener /
        reducer changes the effective focal length.

    pixel_size_um : float
        Physical pixel pitch in micrometres.
        For the ASI533MC: 3.76

    central_obstruction : float
        Fractional linear central obstruction  (secondary / primary diameter).
        0.0 for a refractor (no obstruction).
        Typical Newtonian: 0.2 – 0.35.

    qe_weighted_wavelength_nm : float
        Effective wavelength used for Airy disk computation.
        For broadband use a QE-weighted mean (~530 nm for a typical CMOS).
        For narrowband use the filter's central wavelength.
    """
    aperture_mm:               float = 100.0
    focal_length_mm:           float = 550.0
    pixel_size_um:             float = 3.76
    central_obstruction:       float = 0.0
    qe_weighted_wavelength_nm: float = 530.0

    @property
    def plate_scale_arcsec_per_px(self) -> float:
        """Nominal plate scale from scope geometry [arcsec/pixel]."""
        # pixel_size / focal_length in radians, converted to arcsec
        return (self.pixel_size_um * 1e-3 / self.focal_length_mm) * (180 / np.pi) * 3600

    @property
    def f_number(self) -> float:
        return self.focal_length_mm / self.aperture_mm

    @property
    def airy_radius_arcsec(self) -> float:
        """Angular radius of Airy first dark ring [arcsec]."""
        lam_mm = self.qe_weighted_wavelength_nm * 1e-6
        return 1.22 * (lam_mm / self.aperture_mm) * (180 / np.pi) * 3600

    @property
    def airy_radius_pixels(self) -> float:
        """Airy first dark ring radius in pixels (nominal plate scale)."""
        return self.airy_radius_arcsec / self.plate_scale_arcsec_per_px

    def summary(self) -> str:
        lines = [
            f"Aperture          : {self.aperture_mm:.1f} mm",
            f"Focal length      : {self.focal_length_mm:.1f} mm",
            f"f/ratio           : f/{self.f_number:.1f}",
            f"Pixel size        : {self.pixel_size_um:.2f} µm",
            f"Obstruction       : {self.central_obstruction:.2f}",
            f"Wavelength (eff.) : {self.qe_weighted_wavelength_nm:.0f} nm",
            f"Plate scale       : {self.plate_scale_arcsec_per_px:.3f} arcsec/px",
            f"Airy radius       : {self.airy_radius_arcsec:.3f} arcsec"
                f"  ({self.airy_radius_pixels:.2f} px)",
        ]
        return "\n".join(lines)


# ============================================================================
# Airy disk PSF
# ============================================================================

def compute_airy_psf(
    geometry: ScopeGeometry,
    kernel_size: int = 64,
    wavelength_nm: Optional[float] = None,
    plate_scale_arcsec_per_px: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the diffraction-limited PSF (Airy pattern) on the sensor
    pixel grid.

    For an unobstructed circular aperture:

        I(r) = [ 2 J₁(π D r / λ f) / (π D r / λ f) ]²

    For an annular aperture with fractional obstruction ε:

        I(r) = [ 2 J₁(u) / u  −  ε² · 2 J₁(εu) / (εu) ]²  /  (1 − ε²)²

        where  u = π D r / (λ f)

    This is the exact scalar diffraction result for a circular aperture
    with a concentric circular secondary.

    Parameters
    ----------
    geometry : ScopeGeometry
        Scope and camera parameters.

    kernel_size : int
        Output kernel side length in pixels.  Should be odd for a
        centred kernel; even values are accepted (centre at pixel
        kernel_size//2).  64 is more than sufficient for typical
        seeing-dominated setups; use 128 for very good seeing or
        high-magnification rigs.

    wavelength_nm : float, optional
        Override the geometry's qe_weighted_wavelength_nm.
        Useful for computing per-filter PSFs.

    plate_scale_arcsec_per_px : float, optional
        Override the nominal plate scale with the WCS-measured value.
        Always prefer the plate-solved value when available.

    Returns
    -------
    psf : np.ndarray, shape (kernel_size, kernel_size), float64
        Normalised PSF kernel — sums to 1.0.
        Centre of the Airy disk is at pixel (kernel_size//2, kernel_size//2).
    """
    lam_nm  = wavelength_nm if wavelength_nm is not None \
              else geometry.qe_weighted_wavelength_nm
    ps      = plate_scale_arcsec_per_px if plate_scale_arcsec_per_px is not None \
              else geometry.plate_scale_arcsec_per_px

    lam_mm  = lam_nm * 1e-6          # nm → mm
    D       = geometry.aperture_mm
    f       = geometry.focal_length_mm
    eps     = geometry.central_obstruction   # fractional linear obstruction

    # Pixel coordinates relative to kernel centre
    cx, cy  = kernel_size // 2, kernel_size // 2
    y, x    = np.mgrid[0:kernel_size, 0:kernel_size]
    r_pix   = np.sqrt((x - cx)**2.0 + (y - cy)**2.0)   # [pixels]

    # Convert pixel radius to physical radius on focal plane [mm]
    pixel_size_mm = geometry.pixel_size_um * 1e-3
    r_mm    = r_pix * pixel_size_mm

    # Dimensionless argument of the Bessel function
    # u = π D r / (λ f)
    u = np.pi * D * r_mm / (lam_mm * f)

    # Airy function A(u) = 2 J₁(u) / u,  A(0) = 1 by L'Hôpital
    def _airy(u: np.ndarray) -> np.ndarray:
        out = np.ones_like(u)
        mask = u > 1e-9
        out[mask] = 2.0 * j1(u[mask]) / u[mask]
        return out

    if eps == 0.0:
        # Unobstructed — simple Airy pattern
        psf = _airy(u) ** 2
    else:
        # Annular aperture — central obstruction ε
        # Normalisation factor (1 − ε²)² preserves total energy = 1
        term = _airy(u) - eps**2 * _airy(eps * u)
        psf  = (term / (1.0 - eps**2)) ** 2

    # Normalise to sum = 1
    total = psf.sum()
    if total > 0:
        psf /= total
    else:
        raise RuntimeError("Airy PSF sums to zero — check scope parameters.")

    logger.debug(
        "Airy PSF: λ=%.0fnm  D=%.0fmm  f=%.0fmm  ε=%.2f  "
        "Airy radius=%.2f px  kernel=%dx%d  peak=%.4f",
        lam_nm, D, f, eps,
        geometry.airy_radius_pixels, kernel_size, kernel_size, psf.max()
    )

    return psf


def compute_broadband_psf(
    geometry: ScopeGeometry,
    wavelengths_nm: np.ndarray,
    weights: np.ndarray,
    kernel_size: int = 64,
    plate_scale_arcsec_per_px: Optional[float] = None,
) -> np.ndarray:
    """
    Compute a wavelength-integrated (broadband) Airy PSF.

    The PSF is integrated over the supplied wavelength grid, weighted by
    `weights` (typically sensor QE × filter transmission × atmosphere).

    Parameters
    ----------
    wavelengths_nm : array (N,)
        Wavelength sample points [nm].
    weights : array (N,)
        Weight at each wavelength (need not be normalised).

    Returns
    -------
    psf : np.ndarray (kernel_size, kernel_size), float64, sums to 1.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()   # normalise

    psf = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    for lam, w in zip(wavelengths_nm, weights):
        psf += w * compute_airy_psf(
            geometry, kernel_size=kernel_size,
            wavelength_nm=float(lam),
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        )
    psf /= psf.sum()
    return psf


# ============================================================================
# WCS geometry extraction
# ============================================================================

@dataclass
class WCSGeometry:
    """
    Camera geometry extracted from a WCS (World Coordinate System) solution.

    Produced by `extract_wcs_geometry()`.  Typically obtained from a
    plate-solve of a light frame.

    Attributes
    ----------
    position_angle_deg : float
        Rotation of the camera's +Y axis (columns) relative to celestial
        North, measured East of North [degrees].
        0° = North up, East left.
        90° = East up, North right.

    plate_scale_arcsec_per_px : float
        Measured plate scale [arcsec/pixel].  Prefer this over the nominal
        value computed from focal length and pixel size — it captures the
        effect of flatteners, reducers, and temperature-induced focus shift.

    flip_x : bool
        True if the X axis is mirrored (e.g. some diagonal configurations).

    flip_y : bool
        True if the Y axis is mirrored.

    ra_centre_deg : float
        Right Ascension of the frame centre [degrees].

    dec_centre_deg : float
        Declination of the frame centre [degrees].

    wcs : astropy.wcs.WCS
        The full WCS object for arbitrary pixel ↔ sky transformations.
    """
    position_angle_deg:       float
    plate_scale_arcsec_per_px: float
    flip_x:                   bool
    flip_y:                   bool
    ra_centre_deg:            float
    dec_centre_deg:           float
    wcs:                      WCS

    def summary(self) -> str:
        flip = []
        if self.flip_x: flip.append("X")
        if self.flip_y: flip.append("Y")
        flip_str = ", ".join(flip) if flip else "none"
        return (
            f"Position angle    : {self.position_angle_deg:.2f}°\n"
            f"Plate scale       : {self.plate_scale_arcsec_per_px:.4f} arcsec/px\n"
            f"Flip              : {flip_str}\n"
            f"Field centre      : RA={self.ra_centre_deg:.4f}°  "
                f"Dec={self.dec_centre_deg:.4f}°"
        )


def extract_wcs_geometry(header: fits.Header) -> WCSGeometry:
    """
    Extract camera geometry from a FITS header containing a WCS solution.

    Supports both CD matrix (CD1_1 etc.) and CROTA2 / CDELT forms.
    The CD matrix is preferred — it is more general and is what modern
    plate solvers (ASTAP, Astrometry.net) produce.

    CD matrix convention
    --------------------
    The CD matrix maps pixel offsets to sky offsets (in degrees):

        Δα cos(δ)   =  CD1_1 · Δx  +  CD1_2 · Δy
        Δδ          =  CD2_1 · Δx  +  CD2_2 · Δy

    From this we extract:

        plate_scale = sqrt(CD1_1² + CD2_1²) · 3600  [arcsec/px]
        position_angle = atan2(CD1_2, CD1_1)         [radians → degrees]

    The sign of the determinant det(CD) reveals the flip state:
        det > 0 → standard orientation (RA increases left)
        det < 0 → mirrored (RA increases right, typical for direct-focus)

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header from a plate-solved frame.

    Returns
    -------
    WCSGeometry

    Raises
    ------
    ValueError
        If neither a CD matrix nor CROTA2/CDELT keywords are found.
    """
    wcs = WCS(header, naxis=2)

    # ------------------------------------------------------------------
    # Extract the CD matrix — prefer explicit CD keywords, fall back
    # to the WCS object's computed pixel_scale_matrix
    # ------------------------------------------------------------------
    if all(k in header for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        cd = np.array([
            [header["CD1_1"], header["CD1_2"]],
            [header["CD2_1"], header["CD2_2"]],
        ])
        logger.debug("WCS: using explicit CD matrix from header")

    elif "CDELT1" in header and "CDELT2" in header:
        # CROTA2 / CDELT convention — convert to CD matrix
        cdelt1 = float(header["CDELT1"])
        cdelt2 = float(header["CDELT2"])
        crota2 = float(header.get("CROTA2", 0.0))
        cos_r  = np.cos(np.radians(crota2))
        sin_r  = np.sin(np.radians(crota2))
        cd = np.array([
            [cdelt1 * cos_r,  -cdelt2 * sin_r],
            [cdelt1 * sin_r,   cdelt2 * cos_r],
        ])
        logger.debug("WCS: converted CDELT/CROTA2 to CD matrix")

    else:
        # Last resort — use astropy's pixel_scale_matrix
        try:
            cd = wcs.pixel_scale_matrix
            logger.debug("WCS: using astropy pixel_scale_matrix")
        except Exception as exc:
            raise ValueError(
                "Cannot extract CD matrix from FITS header — "
                "no CD keywords, CDELT, or astropy pixel_scale_matrix "
                f"available. Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Plate scale — quadrature sum of first column of CD matrix
    # Units: degrees/pixel → arcsec/pixel
    # ------------------------------------------------------------------
    plate_scale_deg_per_px = np.sqrt(cd[0, 0]**2 + cd[1, 0]**2)
    plate_scale_arcsec     = plate_scale_deg_per_px * 3600.0

    # ------------------------------------------------------------------
    # Position angle — angle of +Y pixel axis (columns) from North,
    # measured East of North.
    # PA = 0°  → North up (+Y points North, typical equatorial mount)
    # PA = 90° → East up (+Y points East)
    #
    # Derivation from CD matrix:
    #   A unit step in +Y maps to sky offset (CD1_2, CD2_2).
    #   CD1_2 is the RA (East-West) component of that direction.
    #   CD2_2 is the Dec (North-South) component.
    #   PA = atan2(CD1_2, CD2_2)  gives the angle East of North.
    # ------------------------------------------------------------------
    position_angle_deg = float(np.degrees(np.arctan2(cd[0, 1], cd[1, 1])) % 360.0)

    # ------------------------------------------------------------------
    # Flip state — sign of determinant of CD matrix
    # Negative det means one axis is flipped (mirrored image)
    # ------------------------------------------------------------------
    det = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    # Conventional: flip_x True when det > 0 (RA increases to the right)
    flip_x = det > 0
    flip_y = False   # Y flip is less common; would need separate check

    # ------------------------------------------------------------------
    # Field centre — sky coords of the reference pixel (CRPIX)
    # ------------------------------------------------------------------
    crpix1 = float(header.get("CRPIX1", 1.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    sky    = wcs.pixel_to_world(crpix1 - 1, crpix2 - 1)  # 0-indexed
    try:
        ra_deg  = float(sky.ra.deg)
        dec_deg = float(sky.dec.deg)
    except AttributeError:
        # SkyCoord may return different representation
        coords  = sky.icrs
        ra_deg  = float(coords.ra.deg)
        dec_deg = float(coords.dec.deg)

    logger.info(
        "WCS geometry: PA=%.2f°  scale=%.4f\"/px  flip_x=%s  "
        "centre=RA%.4f Dec%.4f",
        position_angle_deg, plate_scale_arcsec, flip_x, ra_deg, dec_deg
    )

    return WCSGeometry(
        position_angle_deg        = position_angle_deg,
        plate_scale_arcsec_per_px = plate_scale_arcsec,
        flip_x                    = flip_x,
        flip_y                    = flip_y,
        ra_centre_deg             = ra_deg,
        dec_centre_deg            = dec_deg,
        wcs                       = wcs,
    )


# ============================================================================
# Sub-pixel shift from WCS pair
# ============================================================================

@dataclass
class FrameShift:
    """
    Sub-pixel translation of a frame relative to a reference frame,
    derived from WCS solutions.

    Using WCS-derived shifts is strictly more accurate than converting
    mount-encoder dither commands — it measures the *actual* executed
    offset including any mount errors.

    Attributes
    ----------
    dx_px, dy_px : float
        Sub-pixel shift in pixels (frame − reference).
        Positive dx = frame is shifted right relative to reference.

    dx_arcsec, dy_arcsec : float
        Same shift expressed in arcsec.

    rotation_deg : float
        Small rotation between frames [degrees].
        Non-zero for alt-az mounts (field rotation) or imperfect polar
        alignment.  For a well-aligned equatorial mount this is ~0.

    scale_ratio : float
        Ratio of plate scales (frame / reference).
        Should be ~1.0; deviation indicates atmospheric dispersion or
        focus change.
    """
    dx_px:        float
    dy_px:        float
    dx_arcsec:    float
    dy_arcsec:    float
    rotation_deg: float
    scale_ratio:  float

    @property
    def shift_magnitude_px(self) -> float:
        return float(np.sqrt(self.dx_px**2 + self.dy_px**2))

    def __repr__(self) -> str:
        return (f"FrameShift(dx={self.dx_px:+.3f}px  dy={self.dy_px:+.3f}px  "
                f"rot={self.rotation_deg:+.4f}°  scale={self.scale_ratio:.5f})")


def compute_frame_shift(
    wcs_frame: WCSGeometry,
    wcs_reference: WCSGeometry,
    frame_shape: Tuple[int, int],
) -> FrameShift:
    """
    Compute the sub-pixel shift of `wcs_frame` relative to `wcs_reference`.

    Uses three reference points (frame centre, centre±offset) to decompose
    the full affine transform (translation + rotation + scale) between the
    two WCS solutions.  For the common case of pure translation (equatorial
    mount with dithering) only dx, dy are significant.

    Parameters
    ----------
    wcs_frame : WCSGeometry
        WCS of the frame to characterise.
    wcs_reference : WCSGeometry
        WCS of the reference frame (typically the first or best-seeing frame).
    frame_shape : (H, W)
        Pixel dimensions of the frame.

    Returns
    -------
    FrameShift
    """
    H, W = frame_shape
    cx, cy = W / 2.0, H / 2.0
    offset = min(H, W) / 4.0

    # Three test points in the FRAME pixel system
    test_px = np.array([
        [cx,          cy],           # centre
        [cx + offset, cy],           # right
        [cx,          cy + offset],  # up
    ])

    # Convert frame pixels → sky → reference pixels for each test point
    ref_px = np.zeros_like(test_px)
    for i, (px, py) in enumerate(test_px):
        # frame pixel → sky (using frame WCS)
        sky = wcs_frame.wcs.pixel_to_world(px, py)
        # sky → reference pixel (using reference WCS)
        ref_x, ref_y = wcs_reference.wcs.world_to_pixel(sky)
        ref_px[i] = [float(ref_x), float(ref_y)]

    # Pure translation from the centre point
    dx_px = float(ref_px[0, 0] - test_px[0, 0])
    dy_px = float(ref_px[0, 1] - test_px[0, 1])

    # Scale ratio from the right-offset point
    d_frame = np.linalg.norm(test_px[1] - test_px[0])
    d_ref   = np.linalg.norm(ref_px[1]  - ref_px[0])
    scale_ratio = float(d_ref / d_frame) if d_frame > 0 else 1.0

    # Rotation angle from the up-offset point
    vec_frame = test_px[2] - test_px[0]
    vec_ref   = ref_px[2]  - ref_px[0]
    angle_frame = float(np.degrees(np.arctan2(vec_frame[1], vec_frame[0])))
    angle_ref   = float(np.degrees(np.arctan2(vec_ref[1],   vec_ref[0])))
    rotation_deg = float(angle_ref - angle_frame)
    # Wrap to [-180, 180]
    rotation_deg = (rotation_deg + 180) % 360 - 180

    # Shift in arcsec
    ps = wcs_reference.plate_scale_arcsec_per_px
    dx_arcsec = dx_px * ps
    dy_arcsec = dy_px * ps

    logger.debug(
        "Frame shift: dx=%.3fpx dy=%.3fpx rot=%.4f° scale=%.5f",
        dx_px, dy_px, rotation_deg, scale_ratio
    )

    return FrameShift(
        dx_px        = dx_px,
        dy_px        = dy_px,
        dx_arcsec    = dx_arcsec,
        dy_arcsec    = dy_arcsec,
        rotation_deg = rotation_deg,
        scale_ratio  = scale_ratio,
    )


# ============================================================================
# ASTAP plate solver wrapper
# ============================================================================

class ASTAPSolver:
    """
    Thin wrapper around the ASTAP command-line plate solver.

    ASTAP is a fast offline solver that writes the WCS solution back into
    the FITS header.  It is the recommended solver for this pipeline
    because it is:
      - Fast enough to run on every light frame without becoming a bottleneck
      - Offline (no network required)
      - Accurate to < 0.1 pixel for typical deep-sky fields

    Installation: https://www.hnsky.org/astap.htm
    Index files must be downloaded separately and placed in ASTAP's star
    catalogue directory.

    Parameters
    ----------
    astap_path : str or Path
        Path to the ASTAP executable.
        Defaults to "astap" (assumes it is on PATH).

    search_radius_deg : float
        Initial search radius for blind solving [degrees].
        Reduce this if the mount's pointing model is good (< 1°).
        Set to ~10° for the first solve of a session without a pointing model.

    downsample : int
        Downsample factor before solving (1 = full resolution).
        2 is a good default — significantly faster with minimal accuracy loss.

    min_stars : int
        Minimum number of matched stars required to accept a solution.
    """

    def __init__(
        self,
        astap_path:        str | Path = "astap",
        search_radius_deg: float      = 5.0,
        downsample:        int        = 2,
        min_stars:         int        = 10,
    ) -> None:
        self.astap_path        = Path(astap_path)
        self.search_radius_deg = search_radius_deg
        self.downsample        = downsample
        self.min_stars         = min_stars

    def solve(
        self,
        fits_path:    str | Path,
        ra_hint_deg:  Optional[float] = None,
        dec_hint_deg: Optional[float] = None,
    ) -> Optional[WCSGeometry]:
        """
        Plate-solve a FITS file and return the WCS geometry.

        ASTAP writes the solution directly into the FITS file's header.
        This method reads the updated header and extracts the WCS geometry.

        Parameters
        ----------
        fits_path : str or Path
            Path to the FITS file to solve.
            The file is modified in-place by ASTAP (WCS keywords added).

        ra_hint_deg, dec_hint_deg : float, optional
            Initial pointing guess [degrees].  Providing these dramatically
            speeds up solving.  Read from the FITS header (OBJCTRA/OBJCTDEC
            or RA/DEC) if not supplied.

        Returns
        -------
        WCSGeometry if solving succeeded, None otherwise.
        """
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        # Try to read RA/Dec hint from header if not supplied
        if ra_hint_deg is None or dec_hint_deg is None:
            with fits.open(fits_path, memmap=False) as hdul:
                header = hdul[0].header
            ra_hint_deg, dec_hint_deg = _read_radec_hint(header)

        # Build ASTAP command
        cmd = [
            str(self.astap_path),
            "-f", str(fits_path),
            "-r", str(self.search_radius_deg),
            "-z", str(self.downsample),
            "-s", str(self.min_stars),
            "-update",          # write solution back into FITS header
        ]
        if ra_hint_deg is not None:
            cmd += ["-ra",  f"{ra_hint_deg / 15.0:.6f}"]   # ASTAP expects hours
        if dec_hint_deg is not None:
            cmd += ["-spd", f"{dec_hint_deg + 90.0:.6f}"]  # south polar distance

        logger.info("ASTAP: solving %s", fits_path.name)
        logger.debug("ASTAP command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output = True,
                text           = True,
                timeout        = 120,     # 2-minute timeout
            )
        except FileNotFoundError:
            logger.error(
                "ASTAP executable not found at '%s'. "
                "Install ASTAP and ensure it is on PATH or supply astap_path.",
                self.astap_path,
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("ASTAP timed out on %s", fits_path.name)
            return None

        if result.returncode != 0:
            logger.warning(
                "ASTAP failed on %s (exit %d):\n%s",
                fits_path.name, result.returncode, result.stderr.strip()
            )
            return None

        # Check that ASTAP actually found a solution
        # ASTAP writes "Solution found" to stdout on success
        if "Solution found" not in result.stdout and \
           "solution found" not in result.stdout.lower():
            logger.warning(
                "ASTAP did not find a solution for %s.\nOutput: %s",
                fits_path.name, result.stdout.strip()
            )
            return None

        # Read the updated header and extract WCS
        try:
            with fits.open(fits_path, memmap=False) as hdul:
                header = hdul[0].header
            geometry = extract_wcs_geometry(header)
            logger.info(
                "ASTAP solved %s: PA=%.1f°  scale=%.4f\"/px",
                fits_path.name,
                geometry.position_angle_deg,
                geometry.plate_scale_arcsec_per_px,
            )
            return geometry
        except Exception as exc:
            logger.error(
                "WCS extraction failed after ASTAP solve of %s: %s",
                fits_path.name, exc
            )
            return None

    def solve_from_header(
        self,
        fits_path: str | Path,
    ) -> Optional[WCSGeometry]:
        """
        Extract WCS geometry from an already-solved FITS file.

        Use this when ASTAP has already been run (e.g. by the capture
        software) and the header already contains a WCS solution.
        No subprocess is invoked.
        """
        fits_path = Path(fits_path)
        with fits.open(fits_path, memmap=False) as hdul:
            header = hdul[0].header
        try:
            return extract_wcs_geometry(header)
        except ValueError as exc:
            logger.warning(
                "No WCS solution found in %s: %s", fits_path.name, exc
            )
            return None


def _read_radec_hint(header: fits.Header) -> Tuple[Optional[float], Optional[float]]:
    """Try to read RA/Dec from common FITS header keywords."""
    ra, dec = None, None

    # Try numeric keywords first (degrees)
    for kw in ("RA", "OBJCTRA", "TELRA", "RA_OBJ"):
        v = header.get(kw)
        if v is not None:
            try:
                val = float(v)
                # OBJCTRA is often in hours ("HH MM SS") or degrees
                # Simple heuristic: if string and has spaces, parse as HMS
                if isinstance(v, str) and " " in v.strip():
                    from astropy.coordinates import Angle
                    import astropy.units as u
                    ra = Angle(v.strip(), unit=u.hourangle).deg
                elif val < 24.0 and isinstance(v, str):
                    # Probably hours
                    ra = val * 15.0
                else:
                    ra = val
                break
            except Exception:
                pass

    for kw in ("DEC", "OBJCTDEC", "TELDEC", "DEC_OBJ"):
        v = header.get(kw)
        if v is not None:
            try:
                if isinstance(v, str) and (" " in v.strip() or ":" in v):
                    from astropy.coordinates import Angle
                    import astropy.units as u
                    dec = Angle(v.strip(), unit=u.deg).deg
                else:
                    dec = float(v)
                break
            except Exception:
                pass

    return ra, dec


# ============================================================================
# Instrument PSF composition
# ============================================================================

def get_instrument_psf(
    geometry:      ScopeGeometry,
    psf_optics:    Optional[np.ndarray] = None,
    kernel_size:   int                  = 64,
    wavelength_nm: Optional[float]      = None,
    plate_scale_arcsec_per_px: Optional[float] = None,
) -> np.ndarray:
    """
    Return the best available instrument PSF kernel.

    Priority
    --------
    1. H_diff * H_optics  — full model (when psf_optics is provided)
    2. H_diff             — analytic Airy only (psf_optics is None)

    The result is used as a fixed kernel in the MAP stacker forward model.
    H_seeing is NOT included here — it varies per frame and is handled by
    FrameCharacterizer.

    Parameters
    ----------
    geometry : ScopeGeometry
        Scope and camera parameters.
    psf_optics : np.ndarray (K, K), optional
        Optical aberration PSF estimated from data.  None = not yet
        calibrated; Airy only is used.
    kernel_size : int
        Kernel output size.  Must match psf_optics size if supplied.
    wavelength_nm : float, optional
        Override wavelength.
    plate_scale_arcsec_per_px : float, optional
        Override plate scale (use WCS-measured value when available).

    Returns
    -------
    psf : np.ndarray (kernel_size, kernel_size), float64, sums to 1.
    """
    psf_diff = compute_airy_psf(
        geometry,
        kernel_size                = kernel_size,
        wavelength_nm              = wavelength_nm,
        plate_scale_arcsec_per_px  = plate_scale_arcsec_per_px,
    )

    if psf_optics is None:
        logger.info(
            "get_instrument_psf: psf_optics not calibrated — "
            "using diffraction PSF only (good approximation for Esprit 100ED)"
        )
        return psf_diff

    # Convolve in Fourier space: H_instrument = H_diff * H_optics
    if psf_optics.shape != psf_diff.shape:
        raise ValueError(
            f"psf_optics shape {psf_optics.shape} != "
            f"psf_diff shape {psf_diff.shape}"
        )
    from numpy.fft import rfft2, irfft2
    psf_instrument = irfft2(rfft2(psf_diff) * rfft2(psf_optics))
    psf_instrument = np.real(psf_instrument)
    psf_instrument = np.maximum(psf_instrument, 0.0)
    psf_instrument /= psf_instrument.sum()

    logger.info("get_instrument_psf: using H_diff * H_optics")
    return psf_instrument
