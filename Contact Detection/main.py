"""Utilities for detecting contact regions from 3D point clouds."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EllipsoidFit:
	"""Geometric parameters of a fitted ellipsoid."""

	center: np.ndarray
	radii: np.ndarray
	rotation: np.ndarray

@dataclass(frozen=True)
class PlaneFit:
	"""Geometric parameters of a fitted plane."""

	point: np.ndarray
	normal: np.ndarray

SurfaceFit = EllipsoidFit | PlaneFit


def signed_distance_to_surface(
	point: np.ndarray,
	surface: SurfaceFit,
) -> float:
	"""Return a signed distance or a fast radial distance approximation.

	``point`` must have shape ``(3,)``.
	For a plane this is the exact signed perpendicular distance. For an
	ellipsoid, the sign is exact and the magnitude is a radial approximation.
	"""
	point_array = np.asarray(point, dtype=float)

	if point_array.shape != (3,):
		raise ValueError("point must have shape (3,)")

	if isinstance(surface, PlaneFit):
		return float((point_array - surface.point) @ surface.normal)
	else:
		local = (point_array - surface.center) @ surface.rotation / surface.radii
		normalized_radius = np.linalg.norm(local, axis=-1)
		return float((normalized_radius - 1) * np.min(surface.radii))


def _validate_samples(points: np.ndarray) -> np.ndarray:
	samples = np.asarray(points, dtype=float)
	if samples.ndim != 2 or samples.shape[1] != 3:
		raise ValueError("points must have shape (n, 3)")
	if samples.shape[0] < 3:
		raise ValueError("at least 3 points are required")
	if not np.isfinite(samples).all():
		raise ValueError("points must contain only finite values")
	return samples


def fit_surface(points: np.ndarray) -> SurfaceFit:
	"""Fit and select the better least-squares plane or ellipsoid model.

	The models are compared using RMS signed-distance residuals. Ellipsoid
	 fitting can fail for coplanar or nearly coplanar samples; in that case the
	plane is selected automatically.
	"""
	samples = _validate_samples(points)
	plane = fit_plane(samples)
	plane_residual = np.sqrt(np.mean([
		signed_distance_to_surface(point, plane) ** 2
		for point in samples
	]))
	try:
		ellipsoid = fit_ellipsoid(samples)
		ellipsoid_residual = np.sqrt(np.mean([
			signed_distance_to_surface(point, ellipsoid) ** 2
			for point in samples
		]))
	except ValueError:
		return plane
	return ellipsoid if ellipsoid_residual < plane_residual else plane

def fit_plane(points: np.ndarray) -> PlaneFit:
	"""Fit a least-squares plane to a collection of 3D points."""

	samples = _validate_samples(points)
	_, _, right_singular_vectors = np.linalg.svd(
		samples - samples.mean(axis=0), full_matrices=False
	)
	normal = right_singular_vectors[-1]
	normal /= np.linalg.norm(normal)
	return PlaneFit(point=samples.mean(axis=0), normal=normal)

def fit_ellipsoid(points: np.ndarray) -> EllipsoidFit:
	"""Fit a least-squares ellipsoid through a collection of 3D points.

	``rotation`` contains the principal axes as columns. At least nine
	non-coplanar points are required for a stable solution.
	"""

	samples = _validate_samples(points)
	if samples.shape[0] < 9:
		raise ValueError("at least 9 points are required")

	point_center = samples.mean(axis=0)
	scale = np.linalg.norm(samples - point_center, axis=1).max()
	if scale == 0:
		raise ValueError("points must not all be identical")
	normalized = (samples - point_center) / scale

	x, y, z = normalized.T
	design = np.column_stack((
		x * x,
		y * y,
		z * z,
		2 * x * y,
		2 * x * z,
		2 * y * z,
		2 * x,
		2 * y,
		2 * z,
	))
	coefficients, _, rank, _ = np.linalg.lstsq( design, np.ones(len(samples)), rcond=None)
	if rank < 9:
		raise ValueError("points do not constrain a unique ellipsoid")

	quadratic = np.array([
		[coefficients[0], coefficients[3], coefficients[4]],
		[coefficients[3], coefficients[1], coefficients[5]],
		[coefficients[4], coefficients[5], coefficients[2]], ])
	linear = coefficients[6:9]
	eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
	if np.any(eigenvalues <= 0):
		raise ValueError("the fitted quadratic is not an ellipsoid")

	normalized_center = -np.linalg.solve(quadratic, linear)
	squared_radius_scale = 1 + normalized_center @ quadratic @ normalized_center
	if squared_radius_scale <= 0:
		raise ValueError("the fitted quadratic does not define a real ellipsoid")

	radii = scale * np.sqrt(squared_radius_scale / eigenvalues)
	center = point_center + scale * normalized_center
	order = np.argsort(radii)[::-1]
	return EllipsoidFit(
		center=center,
		radii=radii[order],
		rotation=eigenvectors[:, order],
	)


# TODO: Determine single unique contact moment
# 	- probably the first frame? but I could give a few frames as a buffer to discard erronious keystrokes
# 	- track contact events so future frames are ignored
# 	- Determine which finger contacted
# 	Not sure if its the best place to do it here rather than in the interactive mode functions