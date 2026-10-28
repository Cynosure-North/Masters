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
) -> np.ndarray:
	"""Return the vector from a point to the nearest point on a fitted surface.

	The returned vector is not normalized; its magnitude is the Euclidean
	distance to the closest surface point, and it points from the input point
	oward the surface.
	"""
	point_array = np.asarray(point, dtype=float)

	if point_array.shape != (3,):
		raise ValueError("point must have shape (3,)")

	if isinstance(surface, PlaneFit):
		signed_distance = float((point_array - surface.point) @ surface.normal)
		return -signed_distance * surface.normal
	if isinstance(surface, EllipsoidFit):
		center_to_point = point_array - surface.center
		point_norm = np.linalg.norm(center_to_point)
		if point_norm == 0:
			return np.zeros(3, dtype=float)
		direction = center_to_point / point_norm
		local_direction = direction @ surface.rotation
		scaled_direction = local_direction / surface.radii
		scale = 1.0 / np.linalg.norm(scaled_direction)
		nearest_local = scale * local_direction
		nearest_point = surface.center + nearest_local @ surface.rotation.T
		return nearest_point - point_array
	if isinstance(surface, ConeFit):
		delta = point_array - surface.apex
		axial = float(delta @ surface.axis)
		radial_vector = delta - axial * surface.axis
		radial = np.linalg.norm(radial_vector)
		if radial < 1e-12:
			basis = np.array([1.0, 0.0, 0.0]) if abs(surface.axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
			unit_radial = np.cross(surface.axis, basis)
			if np.linalg.norm(unit_radial) < 1e-12:
				unit_radial = np.cross(surface.axis, np.array([0.0, 0.0, 1.0]))
			unit_radial = unit_radial / np.linalg.norm(unit_radial)
		else:
			unit_radial = radial_vector / radial
		if axial <= 0:
			return surface.apex - point_array
		slope = np.tan(surface.half_angle)
		lambda_value = (axial + slope * radial) / (1.0 + slope * slope)
		nearest_point = surface.apex + lambda_value * surface.axis + slope * lambda_value * unit_radial
		return nearest_point - point_array
	if isinstance(surface, CylinderFit):
		delta = point_array - surface.axis_point
		axial = delta @ surface.axis
		radial_vector = delta - axial * surface.axis
		radial = np.linalg.norm(radial_vector)
		if radial < 1e-12:
			return np.zeros(3, dtype=float)
		unit_radial = radial_vector / radial
		return (surface.radius - radial) * unit_radial
	if isinstance(surface, RbfSurfaceFit):
		def evaluate(current_point: np.ndarray) -> float:
			normalized = (current_point - surface.center) / surface.scale
			distances = np.linalg.norm(surface.centers - normalized, axis=1)
			kernel_value = distances**3 @ surface.weights
			polynomial_value = np.append(1.0, normalized) @ surface.polynomial
			return float((kernel_value + polynomial_value) * surface.scale)
		field_value = evaluate(point_array)
		jacobian = np.empty(3, dtype=float)
		eps = 1e-4
		for axis in range(3):
			delta = np.zeros(3, dtype=float)
			delta[axis] = eps
			jacobian[axis] = (evaluate(point_array + delta) - evaluate(point_array - delta)) / (2.0 * eps)
		gradient_norm = np.linalg.norm(jacobian)
		if gradient_norm < 1e-12:
			return np.zeros(3, dtype=float)
		return -field_value * (jacobian / gradient_norm)



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
			np.linalg.norm(signed_distance_to_surface(point, surface)) ** 2 for point in samples ]))
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

def orthonormal_basis(normal: np.ndarray):
	normal = np.asarray(normal, dtype=np.float32)
	normal = normal / (np.linalg.norm(normal) + 1e-12)
	if abs(normal[0]) < 0.9:
		reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	else:
		reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
	basis_u = np.cross(reference, normal)
	basis_u = basis_u / (np.linalg.norm(basis_u) + 1e-12)
	basis_v = np.cross(normal, basis_u)
	basis_v = basis_v / (np.linalg.norm(basis_v) + 1e-12)
	return basis_u, basis_v


def surface_uv(point: np.ndarray, surface):
	point = np.asarray(point, dtype=np.float32)
	if isinstance(surface, PlaneFit):
		local = point - surface.point
		basis_u, basis_v = orthonormal_basis(surface.normal)
		return np.array([float(local @ basis_u), float(local @ basis_v)], dtype=np.float32)
	if isinstance(surface, CylinderFit):
		centered = point - surface.axis_point
		axial = centered @ surface.axis
		radial = centered - axial * surface.axis
		radial_norm = np.linalg.norm(radial)
		if radial_norm < 1e-12:
			return np.array([float(axial), 0.0], dtype=np.float32)
		basis_u, basis_v = orthonormal_basis(surface.axis)
		angle = np.arctan2(float(radial @ basis_v), float(radial @ basis_u))
		return np.array([float(axial), float(angle)], dtype=np.float32)
	if isinstance(surface, ConeFit):
		delta = point - surface.apex
		axial = float(delta @ surface.axis)
		radial = delta - axial * surface.axis
		radial_norm = np.linalg.norm(radial)
		angle = 0.0 if radial_norm < 1e-12 else np.arctan2(float(radial @ np.cross(surface.axis, np.array([1.0, 0.0, 0.0]))), float(radial @ np.cross(np.array([0.0, 1.0, 0.0]), surface.axis)))
		return np.array([float(axial), float(angle)], dtype=np.float32)
	if isinstance(surface, EllipsoidFit):
		local = (point - surface.center) @ surface.rotation
		return np.array([float(local[0]), float(local[1])], dtype=np.float32)
	if isinstance(surface, RbfSurfaceFit):
		local = point - surface.center
		basis_u, basis_v = orthonormal_basis(local / (np.linalg.norm(local) + 1e-12))
		return np.array([float(local @ basis_u), float(local @ basis_v)], dtype=np.float32)
	return np.array([0.0, 0.0], dtype=np.float32)


def surface_normal(point: np.ndarray, surface):
	point = np.asarray(point, dtype=np.float32)
	if isinstance(surface, PlaneFit):
		return np.asarray(surface.normal, dtype=np.float32) / (np.linalg.norm(surface.normal) + 1e-12)
	if isinstance(surface, CylinderFit):
		delta = point - surface.axis_point
		axial = delta @ surface.axis
		radial = delta - axial * surface.axis
		radial_norm = np.linalg.norm(radial)
		if radial_norm < 1e-12:
			return np.asarray(surface.axis, dtype=np.float32) / (np.linalg.norm(surface.axis) + 1e-12)
		return radial / (radial_norm + 1e-12)
	if isinstance(surface, ConeFit):
		axis = np.asarray(surface.axis, dtype=np.float32)
		axis = axis / (np.linalg.norm(axis) + 1e-12)
		return axis
	if isinstance(surface, EllipsoidFit):
		local = (point - surface.center) @ surface.rotation
		normal = np.asarray(local, dtype=np.float32) / np.maximum(np.linalg.norm(local), 1e-12)
		normal = normal / np.maximum(np.linalg.norm(normal), 1e-12)
		return normal @ surface.rotation.T
	if isinstance(surface, RbfSurfaceFit):
		centered = point - surface.center
		normal = centered / (np.linalg.norm(centered) + 1e-12)
		return np.asarray(normal, dtype=np.float32)
	return np.zeros(3, dtype=np.float32)


def principal_curvatures(point: np.ndarray, surface):
	point = np.asarray(point, dtype=np.float32)
	if isinstance(surface, PlaneFit):
		return np.zeros(2, dtype=np.float32)
	if isinstance(surface, CylinderFit):
		return np.array([0.0, 1.0 / max(surface.radius, 1e-6)], dtype=np.float32)
	if isinstance(surface, ConeFit):
		return np.array([0.0, 1.0 / max(np.linalg.norm(point - surface.apex), 1e-6)], dtype=np.float32)
	if isinstance(surface, EllipsoidFit):
		return np.array([1.0 / max(surface.radii[0], 1e-6), 1.0 / max(surface.radii[1], 1e-6)], dtype=np.float32)
	if isinstance(surface, RbfSurfaceFit):
		return np.array([0.0, 0.0], dtype=np.float32)
	return np.zeros(2, dtype=np.float32)


def principal_curvature_frame(point: np.ndarray, surface):
	"""Return the local surface frame and principal curvature magnitudes.

	The tuple is (normal, tangent_u, tangent_v, curvature_u, curvature_v) where
	tangent_u and tangent_v span the tangent plane, and the curvature values are
	scalar magnitudes rather than direction vectors.
	"""
	point = np.asarray(point, dtype=np.float32)
	normal = surface_normal(point, surface)
	tangent_u, tangent_v = orthonormal_basis(normal)
	curvatures = principal_curvatures(point, surface)
	if isinstance(surface, PlaneFit):
		return normal, tangent_u, tangent_v, curvatures[0], curvatures[1]
	if isinstance(surface, CylinderFit):
		axis = np.asarray(surface.axis, dtype=np.float32)
		axis = axis / (np.linalg.norm(axis) + 1e-12)
		radial = point - surface.axis_point
		radial -= np.dot(radial, axis) * axis
		radial_norm = np.linalg.norm(radial)
		if radial_norm > 1e-12:
			tangent_v = radial / radial_norm
		else:
			tangent_v = tangent_u
		tangent_u = axis
		return normal, tangent_u, tangent_v, 0.0, curvatures[1]
	if isinstance(surface, ConeFit):
		axis = np.asarray(surface.axis, dtype=np.float32)
		axis = axis / (np.linalg.norm(axis) + 1e-12)
		tangent_u = axis
		tangent_v = np.cross(normal, tangent_u)
		if np.linalg.norm(tangent_v) < 1e-12:
			tangent_v = tangent_u
		return normal, tangent_u, tangent_v, 0.0, curvatures[1]
	if isinstance(surface, EllipsoidFit):
		principal_u = np.asarray(surface.rotation[:, 0], dtype=np.float32)
		principal_v = np.asarray(surface.rotation[:, 1], dtype=np.float32)
		return normal, principal_u, principal_v, curvatures[0], curvatures[1]
	return normal, tangent_u, tangent_v, curvatures[0], curvatures[1]
