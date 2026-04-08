"""
star_align.py — catalogue-free star matching and robust similarity fitting.

Public API
----------
match_stars(ref_positions, frame_positions) -> (ref_pts, frame_pts)
fit_similarity_magsac(ref_pts, frame_pts)   -> (dx, dy, rotation_deg, scale) | None

Algorithm
---------
1. Triangle similarity descriptors: for every triple of stars, compute the
   normalised side-ratio vector (a/c, b/c) with a<=b<=c.  This 2-D descriptor
   is invariant to translation, rotation, and uniform scale.
2. KD-tree nearest-neighbour matching on descriptors.
3. Each matched triangle pair yields 3 point correspondences; vote-count to
   find the most consistent point-level matches.
4. MAGSAC++ (cv2.estimateAffinePartial2D with cv2.USAC_MAGSAC) fits the final
   similarity transform (4 DOF: tx, ty, theta, s) without a hard inlier
   threshold.

Dependencies: numpy, scipy, opencv-python-headless (cv2).
Both are optional — callers should check availability before importing.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triangle descriptor helpers
# ---------------------------------------------------------------------------

def _triangle_descriptor(pts: np.ndarray) -> np.ndarray:
    """
    Given a [3, 2] array of (y, x) positions, return the 2-D shape descriptor
    (a/c, b/c) where a<=b<=c are the sorted inter-vertex distances.
    Returns None if the triangle is degenerate (collinear points).
    """
    p0, p1, p2 = pts
    d01 = float(np.linalg.norm(p1 - p0))
    d02 = float(np.linalg.norm(p2 - p0))
    d12 = float(np.linalg.norm(p2 - p1))
    sides = sorted([d01, d02, d12])
    a, b, c = sides
    if c < 1e-6:
        return None
    return np.array([a / c, b / c], dtype=np.float64)


def _build_triangle_catalogue(
    positions: np.ndarray,
    max_stars: int = 50,
) -> Tuple[np.ndarray, list]:
    """
    Build all triangle descriptors for the given star positions (up to
    max_stars brightest).  Returns:
      descriptors : [T, 2] float64 — shape descriptors
      index_triples: list of T (i,j,k) index triples into positions
    """
    pos = positions[:max_stars]
    n = len(pos)
    descriptors = []
    triples = []
    for i, j, k in combinations(range(n), 3):
        pts = pos[[i, j, k]]
        desc = _triangle_descriptor(pts)
        if desc is not None:
            descriptors.append(desc)
            triples.append((i, j, k))
    if not descriptors:
        return np.empty((0, 2)), []
    return np.array(descriptors), triples


# ---------------------------------------------------------------------------
# Public: match_stars
# ---------------------------------------------------------------------------

def match_stars(
    ref_positions:   np.ndarray,
    frame_positions: np.ndarray,
    max_stars:       int   = 50,
    desc_tol:        float = 0.02,
    min_vote:        int   = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find corresponding star positions between a reference frame and a target
    frame using triangle similarity matching.

    Parameters
    ----------
    ref_positions   : [N, 2] (y, x) star positions in the reference frame
    frame_positions : [M, 2] (y, x) star positions in the target frame
    max_stars       : use at most this many stars from each frame (brightest first)
    desc_tol        : max Euclidean distance between descriptors to accept a match
    min_vote        : minimum number of triangle votes for a point pair to be
                      included in the output

    Returns
    -------
    ref_pts   : [K, 2] matched reference positions
    frame_pts : [K, 2] matched frame positions
    Both arrays are empty [0, 2] if fewer than 3 correspondences are found.
    """
    if len(ref_positions) < 3 or len(frame_positions) < 3:
        return np.empty((0, 2)), np.empty((0, 2))

    ref_desc,   ref_trips   = _build_triangle_catalogue(ref_positions,   max_stars)
    frame_desc, frame_trips = _build_triangle_catalogue(frame_positions, max_stars)

    if len(ref_desc) == 0 or len(frame_desc) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # KD-tree query: for each frame triangle find the nearest reference triangle
    tree = cKDTree(ref_desc)
    dists, idxs = tree.query(frame_desc, k=1, workers=1)

    # Vote: each matched triangle pair casts votes for its 3 point pairs
    # votes[(ref_idx, frame_idx)] += 1
    votes: dict = {}
    for fi, (dist, ri) in enumerate(zip(dists, idxs)):
        if dist > desc_tol:
            continue
        r_trip = ref_trips[ri]
        f_trip = frame_trips[fi]
        for r_pt, f_pt in zip(r_trip, f_trip):
            key = (r_pt, f_pt)
            votes[key] = votes.get(key, 0) + 1

    # Keep pairs with enough votes
    accepted = [(r, f) for (r, f), v in votes.items() if v >= min_vote]
    if len(accepted) < 3:
        logger.debug("match_stars: only %d pairs with >=%d votes", len(accepted), min_vote)
        return np.empty((0, 2)), np.empty((0, 2))

    ref_pos   = ref_positions[:max_stars]
    frame_pos = frame_positions[:max_stars]
    ref_pts   = np.array([ref_pos[r]   for r, _ in accepted])
    frame_pts = np.array([frame_pos[f] for _, f in accepted])

    logger.debug("match_stars: %d correspondences found", len(ref_pts))
    return ref_pts, frame_pts


# ---------------------------------------------------------------------------
# Public: fit_similarity_magsac
# ---------------------------------------------------------------------------

def fit_similarity_magsac(
    ref_pts:   np.ndarray,
    frame_pts: np.ndarray,
    reproj_threshold: float = 5.0,
    max_iters:        int   = 2000,
    confidence:       float = 0.999,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit a similarity transform (tx, ty, rotation, scale) from frame coords to
    reference coords using MAGSAC++ via OpenCV.

    Parameters
    ----------
    ref_pts   : [K, 2] (y, x) in reference frame — destination
    frame_pts : [K, 2] (y, x) in target frame   — source
    reproj_threshold : starting noise scale for MAGSAC++ (pixels)

    Returns
    -------
    (dx_px, dy_px, rotation_deg, scale_ratio) or None if fit fails.
    dx_px, dy_px : translation from frame to reference (in reference pixels)
    rotation_deg : CCW rotation of frame relative to reference
    scale_ratio  : frame pixel scale / reference pixel scale
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not available — cannot run MAGSAC++ fit")
        return None

    if len(ref_pts) < 3 or len(frame_pts) < 3:
        return None

    # OpenCV uses (x, y) convention; our arrays are (y, x) — swap columns
    src = frame_pts[:, ::-1].astype(np.float32).reshape(-1, 1, 2)  # frame (x,y)
    dst = ref_pts[:,   ::-1].astype(np.float32).reshape(-1, 1, 2)  # ref   (x,y)

    # Try USAC_MAGSAC first (superior, no hard threshold); fall back to RANSAC
    # estimateAffinePartial2D only supports RANSAC/LMEDS in some builds.
    M, inlier_mask = None, None
    for method in [getattr(cv2, 'USAC_MAGSAC', None), cv2.RANSAC, cv2.LMEDS]:
        if method is None:
            continue
        try:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                src, dst,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                maxIters=max_iters,
                confidence=confidence,
            )
            break
        except cv2.error:
            continue

    if M is None:
        logger.debug("fit_similarity_magsac: estimateAffinePartial2D returned None")
        return None

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    logger.debug("fit_similarity_magsac: %d/%d inliers", n_inliers, len(ref_pts))

    if n_inliers < 3:
        logger.debug("fit_similarity_magsac: too few inliers (%d)", n_inliers)
        return None

    # M = [[s*cos(θ), -s*sin(θ), tx],
    #      [s*sin(θ),  s*cos(θ), ty]]
    # OpenCV translation is in (x, y); convert to (dy, dx) for our convention
    scale        = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))
    rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    tx           = float(M[0, 2])   # x-translation (cols)
    ty           = float(M[1, 2])   # y-translation (rows)

    # Our convention: dx_px = positive → frame shifted right relative to ref
    # OpenCV dst = M * src, so tx/ty give shift of frame origin in ref coords
    dx_px = tx   # col shift
    dy_px = ty   # row shift

    logger.info(
        "fit_similarity_magsac: dx=%.2fpx dy=%.2fpx rot=%.3fdeg scale=%.5f  (%d inliers)",
        dx_px, dy_px, rotation_deg, scale, n_inliers,
    )
    return dx_px, dy_px, rotation_deg, scale
