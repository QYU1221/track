"""Nonlinear first-detection integration in pixel coordinates.

Integrate N(z; h(x), R) N(x; m, P) using the exact inverse projection and
its area Jacobian. An observation-centered Gaussian is only an integration
proposal, never a second prior. Evidence and conditional moments use the
same weights. Velocity is reconstructed from the original Gaussian's
conditional distribution; one position does not create a measured velocity.
"""
import math
import numpy as np
from ..geometry import quaternion_matrix
from ..geometry_errors import GeometryError
from ..numerical_policy import BIRTH_QUADRATURE_ORDER


_nodes, _weights = np.polynomial.hermite.hermgauss(BIRTH_QUADRATURE_ORDER)
_GRID = np.array([(x, y) for x in _nodes for y in _nodes]) * math.sqrt(2.)
_LOG_WEIGHTS = np.log(np.outer(_weights, _weights).ravel() / math.pi)
_DOMAIN_ERRORS = ('camera_singularity', 'distortion_domain', 'tangent_domain')


def inverse_pixel_geometry(camera, plane, rotation):
    """Return exact positions/Jacobians and a domain mask for pixel batches."""
    if camera.distortion_model == 'none':
        transform = np.vstack((plane.e1, plane.e2, plane.center)) @ rotation @ camera.image_to_camera_rotation @ np.linalg.inv(camera.K)
        intrinsic_inverse = np.linalg.inv(camera.K)

        def inverse(pixels):
            homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
            rays = homogeneous @ transform.T
            normalized = homogeneous @ intrinsic_inverse.T
            with np.errstate(divide='ignore', invalid='ignore'):
                positions = rays[:, :2] / rays[:, 2:]
                jacobians = (transform[:2, :2] - positions[:, :, None] * transform[2, :2]) / rays[:, 2, None, None]
            valid = ((rays[:, 2] > 0) & np.isfinite(positions).all(axis=1)
                     & (np.linalg.norm(positions, axis=1) < math.tan(plane.max_angle_rad))
                     & (np.linalg.norm(normalized[:, :2], axis=1) <= camera.max_normalized_radius))
            return positions, jacobians, valid
    else:
        def inverse(pixels):
            positions = np.zeros((len(pixels), 2))
            jacobians = np.zeros((len(pixels), 2, 2))
            valid = np.zeros(len(pixels), dtype=bool)
            for i, pixel in enumerate(pixels):
                try:
                    ray = camera.pixel_to_camera(pixel)
                    positions[i], derivative = plane.project(rotation @ ray['camera_los'])
                    jacobians[i] = derivative @ rotation @ ray['jacobian_camera_pixel']
                    valid[i] = True
                except GeometryError as error:
                    if error.code not in _DOMAIN_ERRORS:
                        raise
            return positions, jacobians, valid
    return inverse


def nonlinear_birth_updates(state, plane, frame, camera, noise, detections, *, rotation=None):
    """Return (pixel log evidence, conditional Gaussian) for each detection.

Position integration is approximate Gaussian quadrature, not a claim of an
exact Gaussian posterior. The Gaussian velocity conditional is analytic.
"""
    rotation = quaternion_matrix(frame.quaternion_xyzw) if rotation is None else rotation
    inverse = inverse_pixel_geometry(camera, plane, rotation)
    mean = np.asarray(state.mean)
    covariance = np.asarray(state.covariance)
    position_precision = np.linalg.inv(covariance[:2, :2])
    noise = np.asarray(noise)
    noise_precision = np.linalg.inv(noise)
    prior_logdet = np.linalg.slogdet(covariance[:2, :2])[1]
    noise_logdet = np.linalg.slogdet(noise)[1]
    if len(mean) == 4:
        velocity_gain = np.linalg.solve(covariance[:2, :2], covariance[:2, 2:]).T
        conditional_cholesky = np.linalg.cholesky(covariance)[2:, 2:]
        velocity_covariance = conditional_cholesky @ conditional_cholesky.T
    results = []
    for detection in detections:
        observed = np.array([detection.x_px, detection.y_px])
        position, jacobian, valid = inverse(observed[None, :])
        if not valid[0]:
            results.append((-math.inf, None))
            continue
        J = jacobian[0]
        # Local product supplies an efficient proposal even for a narrow prior.
        proposal_covariance = np.linalg.inv(noise_precision + J.T @ position_precision @ J)
        proposal_mean = observed + proposal_covariance @ J.T @ position_precision @ (mean[:2] - position[0])
        proposal_cholesky = np.linalg.cholesky(proposal_covariance)
        pixels = proposal_mean + _GRID @ proposal_cholesky.T
        positions, jacobians, valid = inverse(pixels)
        if not np.any(valid):
            results.append((-math.inf, None))
            continue
        positions = positions[valid]
        prior_residual = positions - mean[:2]
        measurement_residual = pixels[valid] - observed
        log_terms = (_LOG_WEIGHTS[valid]
            - math.log(2 * math.pi) - .5 * (prior_logdet + noise_logdet)
            + np.log(np.diag(proposal_cholesky)).sum()
            - .5 * np.einsum('ni,ij,nj->n', prior_residual, position_precision, prior_residual)
            - .5 * np.einsum('ni,ij,nj->n', measurement_residual, noise_precision, measurement_residual)
            + .5 * np.sum(_GRID[valid] ** 2, axis=1)
            + np.linalg.slogdet(jacobians[valid])[1])
        maximum = float(np.max(log_terms))
        weights = np.exp(log_terms - maximum)
        normalizer = float(weights.sum())
        weights /= normalizer
        position_mean = weights @ positions
        centered = positions - position_mean
        position_covariance = (centered.T * weights) @ centered
        if len(mean) == 4:
            posterior_mean = np.r_[position_mean, mean[2:] + velocity_gain @ (position_mean - mean[:2])]
            cross = position_covariance @ velocity_gain.T
            posterior_covariance = np.block([[position_covariance, cross],
                [cross.T, velocity_covariance + velocity_gain @ position_covariance @ velocity_gain.T]])
        else:
            posterior_mean, posterior_covariance = position_mean, position_covariance
        posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
        posterior = type(state)(tuple(posterior_mean), tuple(map(tuple, posterior_covariance)))
        results.append((maximum + math.log(normalizer), posterior))
    return tuple(results)
