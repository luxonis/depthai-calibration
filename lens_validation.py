"""Geometry-only acceptance checks for OpenCV perspective lens models."""
import cv2
import numpy as np


def validate_perspective_lens(K, D, size, maximum_roundtrip_px=1.0):
    """Sample the whole native sensor, including its boundary and corners.

    Inversion must converge to a unique locally orientation-preserving branch.
    These checks use no depth, target planes, or residual-based masking.
    A sampled audit is a rejection guard, not a proof of global bijectivity.
    """
    K, D = np.asarray(K, float), np.asarray(D, float).reshape(-1)
    w, h = size
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(D).all() or min(K[0, 0], K[1, 1]) <= 0:
        return {'valid': False, 'reason': 'Nonfinite or nonpositive camera model'}
    y, x = np.meshgrid(np.linspace(0, h-1, 101), np.linspace(0, w-1, 161), indexing='ij')
    pixels = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
    rays = cv2.undistortPointsIter(pixels, K, D, np.eye(3), np.eye(3),
                                 (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(-1, 2)

    def forward(q):
        points = np.column_stack([q, np.ones(len(q))])
        return cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)

    projected = forward(rays)
    error = np.linalg.norm(projected - pixels.reshape(-1, 2), axis=1)
    step = 1e-5
    dx = (forward(rays + [step, 0]) - forward(rays - [step, 0])) / (2*step)
    dy = (forward(rays + [0, step]) - forward(rays - [0, step])) / (2*step)
    determinant = dx[:, 0]*dy[:, 1] - dx[:, 1]*dy[:, 0]
    bad_inverse = ~np.isfinite(error) | (error > maximum_roundtrip_px)
    bad_jacobian = ~np.isfinite(determinant) | (determinant <= 0)
    # A rational denominator root between the optical axis and the widest
    # sensor ray invalidates the model even if the sampled inverse jumps it.
    denominator = np.pad(D, (0, max(0, 8-len(D))))[[7, 6, 5, 0]].copy()
    denominator[-1] = 1.
    roots = np.roots(np.trim_zeros(denominator, 'f'))
    radius_squared = float(np.nanmax(np.sum(rays*rays, axis=1)))
    poles = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real <= radius_squared]
    return {'valid': not bool(bad_inverse.any() or bad_jacobian.any() or poles),
            'samples': len(error), 'roundtripFailures': int(bad_inverse.sum()),
            'nonpositiveJacobianSamples': int(bad_jacobian.sum()),
            'maximumRoundtripPx': float(np.max(error)) if np.isfinite(error).all() else None,
            'maximumAllowedRoundtripPx': maximum_roundtrip_px,
            'rationalPolesWithinSensorRadiusSquared': poles,
            'basis': 'Full native sensor grid; sampled lens geometry only'}
