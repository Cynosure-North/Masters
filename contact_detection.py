"""Utilities for detecting contact regions from 3D point clouds."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFit:
	"""Geometric parameters of a fitted plane."""

	point: np.ndarray
	normal: np.ndarray

@dataclass(frozen=True)
class EllipsoidFit:
	"""Geometric parameters of a fitted ellipsoid."""

	center: np.ndarray
	radii: np.ndarray
	rotation: np.ndarray

@dataclass(frozen=True)
class ConeFit:
	"""Geometric parameters of a fitted right circular cone surface."""

	apex: np.ndarray
	axis: np.ndarray
	half_angle: float

@dataclass(frozen=True)
class CylinderFit:
	"""Geometric parameters of a fitted infinite right circular cylinder."""

	axis_point: np.ndarray
	axis: np.ndarray
	radius: float

@dataclass(frozen=True)
class RbfSurfaceFit:
	"""Smooth implicit surface interpolated from a sampled point cloud."""

	center: np.ndarray
	scale: float
	centers: np.ndarray
	weights: np.ndarray
	polynomial: np.ndarray

SurfaceFit = PlaneFit | EllipsoidFit | ConeFit | CylinderFit | RbfSurfaceFit


def signed_distance_to_surface(
	point: np.ndarray,
	surface: SurfaceFit,
) -> float:
	"""Return a signed distance or a fast radial distance approximation.

	``point`` must have shape ``(3,)``.
	For a plane this is the exact signed perpendicular distance.
	For an ellipsoid, the sign is exact and the magnitude is a radial approximation.
	For a cone this is the signed distance to its infinite single-nappe side, with the apex treated as the nearest point behind the cone.
	"""
	point_array = np.asarray(point, dtype=float)

	if point_array.shape != (3,):
		raise ValueError("point must have shape (3,)")

	if isinstance(surface, PlaneFit):
		return float((point_array - surface.point) @ surface.normal)
	if isinstance(surface, EllipsoidFit):
		local = (point_array - surface.center) @ surface.rotation / surface.radii
		normalized_radius = np.linalg.norm(local, axis=-1)
		return float((normalized_radius - 1) * np.min(surface.radii))
	if isinstance(surface, ConeFit):
		delta = point_array - surface.apex
		axial = float(delta @ surface.axis)
		radial_vector = delta - axial * surface.axis
		radial = np.linalg.norm(radial_vector)
		profile_point = np.array([axial, radial])
		generator_direction = np.array([
			np.cos(surface.half_angle),
			np.sin(surface.half_angle) ])
		position = max(float(profile_point @ generator_direction), 0.0)
		distance = np.linalg.norm(profile_point - position * generator_direction)
		inside = axial > 0 and radial < axial * np.tan(surface.half_angle)
		return float(-distance if inside else distance)
	if isinstance(surface, CylinderFit):
		delta = point_array - surface.axis_point
		axial = delta @ surface.axis
		radial = np.linalg.norm(delta - axial * surface.axis)
		return float(radial - surface.radius)
	if isinstance(surface, RbfSurfaceFit):
		normalized = (point_array - surface.center) / surface.scale
		distances = np.linalg.norm(surface.centers - normalized, axis=1)
		kernel_value = distances**3 @ surface.weights
		polynomial_value = np.append(1.0, normalized) @ surface.polynomial
		return float((kernel_value + polynomial_value) * surface.scale)



def _validate_samples(points: np.ndarray, min_samples: int=3, max_samples: int=500) -> np.ndarray:
	if len(points) <= max_samples:
		samples = np.asarray(points, dtype=float)
	else:
		samples = np.random.default_rng(0).shuffle(np.asarray(points, dtype=float))[:max_samples]
	if samples.ndim != 2 or samples.shape[1] != 3:
		raise ValueError("points must have shape (n, 3)")
	if samples.shape[0] < min_samples:
		raise ValueError(f"at least {min_samples} points are required")
	if not np.isfinite(samples).all():
		raise ValueError("points must contain only finite values")
	return samples

def fit_surface(points: np.ndarray) -> SurfaceFit:
	"""Fit and select the best least-squares supported surface model.

	Models are compared using mean squared signed-distance residuals. A model
	is excluded if its fit cannot be determined from the samples. If none of
	the analytic models approximates the samples within 2% of their scale, a
	smooth RBF surface is used as a fallback.
	"""

	samples =_validate_samples(points)
	fits: list[SurfaceFit] = []
	for fit_function in (fit_plane, fit_ellipsoid, fit_cone, fit_cylinder):
		try:
			fits.append(fit_function(samples))
		except ValueError:
			continue
		
	residuals = {
		surface: np.sqrt(np.mean([
			signed_distance_to_surface(point, surface) ** 2 for point in samples ]))
		for surface in fits
	}
	best_fit = min(fits, key=lambda l: residuals[l])
	residual = residuals[best_fit]
	scale = np.linalg.norm(samples - samples.mean(axis=0), axis=1).max()
	if residual <= 0.02 * scale:
		return best_fit
	return fit_rbf_surface(samples)


def fit_rbf_surface(points: np.ndarray) -> RbfSurfaceFit:
	"""Interpolate a smooth implicit surface through a 3D point cloud.

	Local PCA estimates sample normals. An RBF signed-distance field is then
	fitted through the points and small offsets along those normals. Normals
	are oriented away from the cloud centroid, so this fallback is best suited
	to roughly convex, consistently sampled shapes.
	"""
	samples = _validate_samples(points, 6)
	if len(samples) > 160:
		indices = np.linspace(0, len(samples) - 1, 160, dtype=int)
		samples = samples[indices]

	center = samples.mean(axis=0)
	scale = np.linalg.norm(samples - center, axis=1).max()
	if scale == 0:
		raise ValueError("points must not all be identical")
	normalized = (samples - center) / scale
	deltas = normalized[:, None, :] - normalized[None, :, :]
	squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
	neighbor_count = min(12, len(samples) - 1)
	nearest = np.argpartition(squared_distances, neighbor_count, axis=1)[:, :neighbor_count + 1]
	normals = np.empty_like(normalized)
	for index, neighbor_indices in enumerate(nearest):
		neighborhood = normalized[neighbor_indices]
		_, _, vectors = np.linalg.svd(
			neighborhood - neighborhood.mean(axis=0), full_matrices=False
		)
		normal = vectors[-1]
		if normal @ normalized[index] < 0:
			normal = -normal
		elif np.isclose(normal @ normalized[index], 0):
			first_nonzero = np.flatnonzero(np.abs(normal) > 1e-12)
			if len(first_nonzero) and normal[first_nonzero[0]] < 0:
				normal = -normal
		normals[index] = normal

	nearest_distances = np.sqrt(np.partition(squared_distances, 1, axis=1)[:, 1])
	offset = max(float(np.median(nearest_distances)) * 0.25, 1e-4)
	centers = np.vstack((normalized, normalized + offset * normals, normalized - offset * normals))
	values = np.concatenate((
		np.zeros(len(samples)),
		np.full(len(samples), offset),
		np.full(len(samples), -offset) ))
	distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
	kernel = distances**3
	polynomial = np.column_stack((np.ones(len(centers)), centers))
	regularization = np.eye(len(centers)) * 1e-10
	block = np.block([
		[kernel + regularization, polynomial],
		[polynomial.T, np.zeros((polynomial.shape[1], polynomial.shape[1]))] ])
	right_hand_side = np.concatenate((values, np.zeros(polynomial.shape[1])))
	try:
		coefficients = np.linalg.solve(block, right_hand_side)
	except np.linalg.LinAlgError as error:
		raise ValueError("points do not define a stable interpolated surface") from error
	return RbfSurfaceFit(
		center=center,
		scale=float(scale),
		centers=centers,
		weights=coefficients[:len(centers)],
		polynomial=coefficients[len(centers):])

def fit_cylinder(points: np.ndarray) -> CylinderFit:
	"""Fit an infinite right circular cylinder to 3D surface samples.

	PCA estimates the cylinder axis. A least-squares circle fit in the
	perpendicular plane estimates its centerline and radius. At least six
	points and a clearly dominant axial direction are required.
	"""
	samples = _validate_samples(points, 6)
	point_center = samples.mean(axis=0)
	centered = samples - point_center
	_, singular_values, right_singular_vectors = np.linalg.svd(
		centered, full_matrices=False
	)
	if singular_values[0] == 0 or singular_values[0] <= singular_values[1] * 1.05:
		raise ValueError("points do not constrain a cylinder axis")
	axis = right_singular_vectors[0]
	reference = np.eye(3)[np.argmin(np.abs(axis))]
	basis_u = np.cross(axis, reference)
	basis_u /= np.linalg.norm(basis_u)
	basis_v = np.cross(axis, basis_u)
	coordinates = np.column_stack((centered @ basis_u, centered @ basis_v))
	x, y = coordinates.T
	design = np.column_stack((2 * x, 2 * y, np.ones(len(samples))))
	if np.linalg.matrix_rank(design) < 3:
		raise ValueError("points do not constrain a unique cylinder radius")
	circle_parameters, _, _, _ = np.linalg.lstsq(
		design, x * x + y * y, rcond=None
	)
	center_u, center_v, squared_radius_offset = circle_parameters
	squared_radius = squared_radius_offset + center_u**2 + center_v**2
	if squared_radius <= 0:
		raise ValueError("the fitted cylinder radius is not real")
	axis_point = point_center + center_u * basis_u + center_v * basis_v
	return CylinderFit(
		axis_point=axis_point,
		axis=axis,
		radius=float(np.sqrt(squared_radius)) )

def fit_cone(points: np.ndarray) -> ConeFit:
	"""Fit a right cone to 3D points.

	The principal axis is estimated by PCA, then the apex and half-angle are
	fitted from the points' axial and radial distances. At least six points
	with variation along the axis are required.
	"""
	samples = _validate_samples(points, 6)

	point_center = samples.mean(axis=0)
	centered = samples - point_center
	_, singular_values, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
	if singular_values[0] == 0 or singular_values[1] <= singular_values[0] * 1e-10:
		raise ValueError("points do not constrain a cone axis")
	axis = right_singular_vectors[0]
	axial = centered @ axis
	radial = np.linalg.norm(centered - np.outer(axial, axis), axis=1)
	if np.cov(axial, radial)[0, 1] < 0:
		axis = -axis
		axial = -axial

	design = np.column_stack((axial, np.ones(len(samples))))
	if np.linalg.matrix_rank(design) < 2:
		raise ValueError("points do not constrain a cone apex")
	slope, intercept = np.linalg.lstsq(design, radial, rcond=None)[0]
	if slope <= 0:
		raise ValueError("points do not define a single-nappe cone")
	apex = point_center - (intercept / slope) * axis
	return ConeFit(
		apex=apex,
		axis=axis,
		half_angle=float(np.arctan(slope)) )


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

	samples = _validate_samples(points, 9)

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
		2 * z ))
	coefficients, _, rank, _ = np.linalg.lstsq( design, np.ones(len(samples)), rcond=None)
	if rank < 9:
		raise ValueError("points do not constrain a unique ellipsoid")

	quadratic = np.array([
		[coefficients[0], coefficients[3], coefficients[4]],
		[coefficients[3], coefficients[1], coefficients[5]],
		[coefficients[4], coefficients[5], coefficients[2]] ])
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
		rotation=eigenvectors[:, order] )


# TODO: Determine single unique contact moment
# 	- probably the first frame? but I could give a few frames as a buffer to discard erroneous keystrokes
# 	- track contact events so future frames are ignored
# 	- Determine which finger contacted
# 	Not sure if its the best place to do it here rather than in the interactive mode functions
# TODO: Report which hand touched
# TODO: I'll want to store the point of last contact to ensure the keys appear in the right place
# TODO: I'll want to fit different surfaces for each hand
#	Not sure how that will work with the motion model
# TODO: Figure out how to integrate this with the AI
# 	- Convert to UVs?
#		- But that would loose orientation and topology
#	- Measure which shapes fit the most and use those?
#	Whatever I do, it probably makes sense to have a preprocessing layer so I don't have to train a bunch of different models