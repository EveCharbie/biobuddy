import numpy as np
from scipy.optimize import fsolve
from enum import Enum
from abc import ABC, abstractmethod

from ....utils.aliases import Points, points_to_array
from ....utils.checks import check_name


class PathWrapMethod(Enum):
    HYBRID = "hybrid"
    MIDPOINT = "midpoint"
    AXIAL = "axial"


class WrappingObjectReal(ABC):
    def __init__(
        self,
        name: str,
        parent_name: str,
        muscle_name: str = None,
        muscle_group: str = None,
        position: Points = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new wrapping object
        parent_name
            The name of the parent the wrapping object is attached to
        muscle_name
            The name of the muscle that passes along this wrapping object
        muscle_group
            The muscle group the muscle belongs to
        position
            The 3d position of the wrapping object in the local reference frame
        """

        self.name = name
        self.parent_name = check_name(parent_name)
        self.muscle_name = muscle_name
        self.muscle_group = muscle_group
        self.position = position

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def parent_name(self) -> str:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, value: str) -> None:
        self._parent_name = value

    @property
    def muscle_name(self) -> str:
        return self._muscle_name

    @muscle_name.setter
    def muscle_name(self, value: str) -> None:
        self._muscle_name = value

    @property
    def muscle_group(self) -> str:
        return self._muscle_group

    @muscle_group.setter
    def muscle_group(self, value: str) -> None:
        self._muscle_group = value

    @property
    def position(self) -> Points:
        return self._position

    @position.setter
    def position(self, value: Points) -> None:
        self._position = points_to_array(points=value, name="viapoint")

    @abstractmethod
    def to_biomod(self):
        """Define the print function, so it automatically formats things in the file properly."""
        pass


class WrappingEllipsoid(WrappingObjectReal):
    def __init__(
        self,
        name: str,
        parent_name: str,
        muscle_name: str = None,
        muscle_group: str = None,
        position: Points = None,
        rotation: Points = None,
        radii: Points = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new wrapping object
        parent_name
            The name of the parent the wrapping object is attached to
        muscle_name
            The name of the muscle that passes along this wrapping object
        muscle_group
            The muscle group the muscle belongs to
        position
            The 3d position of the wrapping object in the local reference frame
        rotation
            The 3d Euler angles rotating the wrapping object in the right orientation in the local reference frame
        radii
            The 3d radii of the ellipsoid along the x, y and z axis
        """

        super().__init__(
            name=name,
            parent_name=parent_name,
            muscle_name=muscle_name,
            muscle_group=muscle_group,
            position=position,
        )
        self.radii = radii
        self.rotation = rotation



# import numpy as np
# from enum import IntEnum
#
# # Constants from OpenSim
# ELLIPSOID_TOLERANCE_1 = 1e-4
# ELLIPSOID_TINY = 0.00000001
# MU_BLEND_MIN = 0.7073
# MU_BLEND_MAX = 0.9
# NUM_FAN_SAMPLES = 300
# SV_BOUNDARY_BLEND = 0.3
#
# class WrapAction(IntEnum):
#     """Enum for wrap action results"""
#     noWrap = 0
#     insideRadius = 1
#     mandatoryWrap = 2
#
# def EQUAL_WITHIN_ERROR(a, b, tolerance=ELLIPSOID_TOLERANCE_1):
#     """Check if two values are equal within tolerance"""
#     return abs(a - b) < tolerance
#
# def DSIGN(x):
#     """Return sign of x as +1 or -1"""
#     return 1 if x >= 0 else -1
#
# def normalize_or_zero(v):
#     """Normalize vector or return zero if magnitude is too small"""
#     norm = np.linalg.norm(v)
#     if norm < 1e-10:
#         return np.zeros_like(v)
#     return v / norm
#
# class WrapResult:
#     """Container for wrapping results"""
#     def __init__(self):
#         self.r1 = np.zeros(3)
#         self.r2 = np.zeros(3)
#         self.c1 = np.zeros(3)
#         self.sv = np.zeros(3)
#         self.factor = 1.0
#         self.wrap_pts = []
#         self.wrap_path_length = 0.0
#
# class WrapEllipsoid:
#     """
#     Direct translation of OpenSim's WrapEllipsoid class.
#     """
#
#     def __init__(self, dimensions, wrap_sign=0, wrap_axis=0):
#         """
#         Parameters:
#         -----------
#         dimensions : array-like, shape (3,)
#             Ellipsoid radii [a, b, c]
#         wrap_sign : int
#             Sign constraint for wrapping (+1, -1, or 0 for no constraint)
#         wrap_axis : int
#             Axis for wrap constraint (0=x, 1=y, 2=z)
#         """
#         self.dimensions = np.array(dimensions, dtype=float)
#         self._wrapSign = wrap_sign
#         self._wrapAxis = wrap_axis
#
#     def wrap_line(self, point1, point2, previous_wrap=None, method=PathWrapMethod.hybrid):
#         """
#         Calculate the wrapping of one line segment over the ellipsoid.
#         Direct translation of wrapLine() from OpenSim.
#
#         Parameters:
#         -----------
#         point1 : array-like, shape (3,)
#             One end of the line segment
#         point2 : array-like, shape (3,)
#             The other end of the line segment
#         previous_wrap : WrapResult, optional
#             Previous wrap result for continuity
#         method : PathWrapMethod
#             Wrapping method (hybrid, midpoint, or axial)
#
#         Returns:
#         --------
#         action : WrapAction
#             The status of the wrapping
#         result : WrapResult
#             The wrapping results
#         flag : bool
#             Success flag
#         """
#         # Initialize result
#         result = WrapResult()
#         flag = True
#
#         # Copy previous wrap data if available
#         if previous_wrap is not None:
#             result.factor = previous_wrap.factor
#             result.r1 = previous_wrap.r1 * previous_wrap.factor
#             result.r2 = previous_wrap.r2 * previous_wrap.factor
#             result.c1 = previous_wrap.c1.copy()
#             result.sv = previous_wrap.sv.copy()
#
#         # Calculate normalization factor
#         result.factor = 3.0 / np.sum(self.dimensions)
#
#         # Normalize coordinates
#         origin = np.zeros(3)
#         p1 = point1 * result.factor
#         p2 = point2 * result.factor
#         m = origin * result.factor
#         a = self.dimensions * result.factor
#
#         # Check if points are on ellipsoid surface
#         p1e = -1.0
#         p2e = -1.0
#         for i in range(3):
#             p1e += ((p1[i] - m[i]) / a[i])**2
#             p2e += ((p2[i] - m[i]) / a[i])**2
#
#         # Check if points are inside ellipsoid
#         if p1e < -0.0001 or p2e < -0.0001:
#             flag = False
#             result.wrap_path_length = 0.0
#             result.r1 /= result.factor
#             result.r2 /= result.factor
#             return WrapAction.insideRadius, result, flag
#
#         # Calculate vectors
#         p1p2 = p1 - p2
#         p1m = normalize_or_zero(p1 - m)
#         p2m = normalize_or_zero(p2 - m)
#
#         ppm = np.dot(p1m, p2m) - 1.0
#
#         # Check collinearity
#         if abs(ppm) < 0.0001:
#             flag = False
#             result.wrap_path_length = 0.0
#             result.r1 /= result.factor
#             result.r2 /= result.factor
#             return WrapAction.noWrap, result, flag
#
#         # Check line-ellipsoid intersection
#         f1 = p1p2 / a
#         f2 = (p2 - m) / a
#         aa = np.dot(f1, f1)
#         bb = 2.0 * np.dot(f1, f2)
#         cc = np.dot(f2, f2) - 1.0
#         disc = bb**2 - 4.0 * aa * cc
#
#         if disc < 0.0:
#             flag = False
#             result.wrap_path_length = 0.0
#             result.r1 /= result.factor
#             result.r2 /= result.factor
#             return WrapAction.noWrap, result, flag
#
#         l1 = (-bb + np.sqrt(disc)) / (2.0 * aa)
#         l2 = (-bb - np.sqrt(disc)) / (2.0 * aa)
#
#         if not (0.0 < l1 < 1.0) or not (0.0 < l2 < 1.0):
#             flag = False
#             result.wrap_path_length = 0.0
#             result.r1 /= result.factor
#             result.r2 /= result.factor
#             return WrapAction.noWrap, result, flag
#
#         # Calculate intersection points
#         result.r1 = p2 + l1 * p1p2
#         result.r2 = p2 + l2 * p1p2
#
#         # ==== COMPUTE WRAPPING PLANE ====
#         r1r2 = result.r2 - result.r1
#         mu = normalize_or_zero(p1p2)
#         mu = np.abs(mu)
#
#         # Arrays for different wrapping approaches
#         t = np.zeros(3)
#         t_sv = np.zeros((3, 3))
#         t_c1 = np.zeros((3, 3))
#
#         # Calculate for each axis
#         for i in range(3):
#             t[i] = (m[i] - result.r1[i]) / r1r2[i]
#             t_sv[i] = result.r1 + t[i] * r1r2
#             t_c1[i] = self.find_closest_point(a[0], a[1], a[2],
#                                              t_sv[i][0], t_sv[i][1], t_sv[i][2], i)
#
#         # Pick most parallel major axis
#         bestMu = np.argmax(mu)
#
#         fanWeight = -np.inf
#
#         if method == PathWrapMethod.hybrid or method == PathWrapMethod.axial:
#             # Frans technique with blending
#             if method == PathWrapMethod.hybrid and mu[bestMu] > MU_BLEND_MIN:
#                 s = 1.0
#                 if t[bestMu] < 0.0 or t[bestMu] > 1.0:
#                     s = 0.0
#                 elif t[bestMu] < SV_BOUNDARY_BLEND:
#                     s = t[bestMu] / SV_BOUNDARY_BLEND
#                 elif t[bestMu] > (1.0 - SV_BOUNDARY_BLEND):
#                     s = (1.0 - t[bestMu]) / SV_BOUNDARY_BLEND
#
#                 if s < 1.0:
#                     mu[bestMu] = MU_BLEND_MIN + s * (mu[bestMu] - MU_BLEND_MIN)
#
#             if method == PathWrapMethod.axial or mu[bestMu] > MU_BLEND_MIN:
#                 result.c1 = t_c1[bestMu]
#                 result.sv = t_sv[bestMu]
#
#             if method == PathWrapMethod.hybrid and mu[bestMu] < MU_BLEND_MAX:
#                 # Fan technique
#                 v_sum = np.zeros(3)
#                 t_sv[2] = result.r1 + 0.5 * r1r2
#
#                 for i in range(1, NUM_FAN_SAMPLES - 1):
#                     tt = i / NUM_FAN_SAMPLES
#                     t_sv[0] = result.r1 + tt * r1r2
#                     t_c1[0] = self.find_closest_point(a[0], a[1], a[2],
#                                                      t_sv[0][0], t_sv[0][1], t_sv[0][2])
#                     v = t_c1[0] - t_sv[0]
#                     v = normalize_or_zero(v)
#                     v_sum += v
#
#                 v_sum = normalize_or_zero(v_sum)
#                 t_c1[0] = t_sv[2] + v_sum
#
#                 if mu[bestMu] <= MU_BLEND_MIN:
#                     result.c1 = self.find_closest_point(a[0], a[1], a[2],
#                                                         t_c1[0][0], t_c1[0][1], t_c1[0][2])
#                     result.sv = t_sv[2]
#                     fanWeight = 1.0
#                 else:
#                     tt = (mu[bestMu] - MU_BLEND_MIN) / (MU_BLEND_MAX - MU_BLEND_MIN)
#                     oneMinusT = 1.0 - tt
#                     t_c1[1] = self.find_closest_point(a[0], a[1], a[2],
#                                                      t_c1[0][0], t_c1[0][1], t_c1[0][2])
#                     t_c1[2] = tt * result.c1 + oneMinusT * t_c1[1]
#                     result.sv = tt * result.sv + oneMinusT * t_sv[2]
#                     result.c1 = self.find_closest_point(a[0], a[1], a[2],
#                                                         t_c1[2][0], t_c1[2][1], t_c1[2][2])
#                     fanWeight = oneMinusT
#         else:  # midpoint method
#             result.sv = result.r1 + 0.5 * (result.r2 - result.r1)
#             result.c1 = self.find_closest_point(a[0], a[1], a[2],
#                                                result.sv[0], result.sv[1], result.sv[2])
#
#         # Initialize tangent points from c1
#         use_c1_to_find_tangent_pts = True
#         if method == PathWrapMethod.axial:
#             use_c1_to_find_tangent_pts = (0.0 < t[bestMu] < 1.0)
#
#         if use_c1_to_find_tangent_pts:
#             result.r1 = result.c1.copy()
#             result.r2 = result.c1.copy()
#
#         # Handle wrap sign constraint
#         if self._wrapSign != 0:
#             dist = result.c1[self._wrapAxis] - m[self._wrapAxis]
#             if DSIGN(dist) != self._wrapSign:
#                 orig_c1 = result.c1.copy()
#                 result.c1[self._wrapAxis] = -result.c1[self._wrapAxis]
#                 result.r1 = result.c1.copy()
#                 result.r2 = result.c1.copy()
#
#                 if EQUAL_WITHIN_ERROR(fanWeight, -np.inf):
#                     fanWeight = 1.0 - (mu[bestMu] - MU_BLEND_MIN) / (MU_BLEND_MAX - MU_BLEND_MIN)
#
#                 fanWeight = min(fanWeight, 1.0)
#
#                 if fanWeight > 0.0:
#                     bisection = (orig_c1[self._wrapAxis] + result.c1[self._wrapAxis]) / 2.0
#                     result.c1[self._wrapAxis] += fanWeight * (bisection - result.c1[self._wrapAxis])
#                     tc1 = result.c1.copy()
#                     result.c1 = self.find_closest_point(a[0], a[1], a[2],
#                                                         tc1[0], tc1[1], tc1[2])
#
#         # Calculate wrapping plane
#         p1c1 = p1 - result.c1
#         vs = np.cross(p1p2, p1c1)
#         vs = normalize_or_zero(vs)
#         vs4 = -np.dot(vs, result.c1)
#
#         # Find tangent points
#         self.calc_tangent_point(p1e, result.r1, p1, m, a, vs, vs4)
#         self.calc_tangent_point(p2e, result.r2, p2, m, a, vs, vs4)
#
#         # Calculate path on ellipsoid
#         far_side_wrap = False
#         self.calc_distance_on_ellipsoid(result.r1, result.r2, m, a, vs, vs4,
#                                        far_side_wrap, result)
#
#         # Check for wrong-way wrap
#         if self._wrapSign != 0 and len(result.wrap_pts) > 2 and not far_side_wrap:
#             w1 = result.wrap_pts[1]
#             w2 = result.wrap_pts[-2]
#
#             r1p1 = normalize_or_zero(p1 - result.r1)
#             r1w1 = normalize_or_zero(w1 - result.r1)
#             r2p2 = normalize_or_zero(p2 - result.r2)
#             r2w2 = normalize_or_zero(w2 - result.r2)
#
#             if np.dot(r1p1, r1w1) > 0.0 or np.dot(r2p2, r2w2) > 0.0:
#                 far_side_wrap = True
#                 self.calc_distance_on_ellipsoid(result.r1, result.r2, m, a, vs, vs4,
#                                                far_side_wrap, result)
#
#         # Unfactor coordinates
#         result.wrap_path_length /= result.factor
#         result.wrap_pts = [pt / result.factor for pt in result.wrap_pts]
#         result.r1 /= result.factor
#         result.r2 /= result.factor
#
#         return WrapAction.mandatoryWrap, result, flag
#
#     def calc_tangent_point(self, p1e, r1, p1, m, a, vs, vs4):
#         """Calculate tangent point. Direct translation from OpenSim."""
#         if abs(p1e) < 0.0001:
#             r1[:] = p1
#             return 1
#
#         nr1 = 2.0 * (r1 - m) / (a**2)
#         d1 = -np.dot(nr1, r1)
#
#         ee = np.zeros(4)
#         ee[0] = np.dot(vs, r1) + vs4
#         ee[1] = -1.0 + np.sum(((r1 - m) / a)**2)
#         ee[2] = np.dot(nr1, r1) + d1
#         ee[3] = np.dot(nr1, p1) + d1
#
#         ssqo = np.sum(ee**2)
#         ssq = ssqo
#
#         nit = 0
#         maxit = 50
#         maxit2 = 1000
#         alpha = 0.01
#
#         while ssq > ELLIPSOID_TINY and nit < maxit:
#             nit += 1
#
#             # Build Jacobian
#             dedth = np.zeros((4, 4))
#             for i in range(3):
#                 dedth[i, 0] = vs[i]
#                 dedth[i, 1] = 2.0 * (r1[i] - m[i]) / a[i]**2
#                 dedth[i, 2] = 2.0 * (2.0 * r1[i] - m[i]) / a[i]**2
#                 dedth[i, 3] = 2.0 * p1[i] / a[i]**2
#             dedth[3, 0] = 0.0
#             dedth[3, 1] = 0.0
#             dedth[3, 2] = 1.0
#             dedth[3, 3] = 1.0
#
#             p1r1 = normalize_or_zero(p1 - r1)
#             p1m = normalize_or_zero(p1 - m)
#             pcos = np.dot(p1r1, p1m)
#
#             dd = 1.0 - pow(pcos, 100) if pcos > 0.1 else 1.0
#
#             v = -dedth.T @ ee
#             dedth2 = dedth.T @ dedth
#             diag = np.diag(dedth2).copy()
#
#             nit2 = 0
#             while ssq >= ssqo and nit2 < maxit2:
#                 dedth2_mod = dedth2.copy()
#                 for i in range(4):
#                     dedth2_mod[i, i] = diag[i] * (1.0 + alpha)
#
#                 ddinv2 = np.linalg.inv(dedth2_mod)
#                 vt = dd * ddinv2 @ v / 16.0
#
#                 r1[:3] += vt[:3]
#                 d1 += vt[3]
#
#                 nr1 = 2.0 * (r1 - m) / (a**2)
#                 ee[0] = np.dot(vs, r1) + vs4
#                 ee[1] = -1.0 + np.sum(((r1 - m) / a)**2)
#                 ee[2] = np.dot(nr1, r1) + d1
#                 ee[3] = np.dot(nr1, p1) + d1
#
#                 ssqo = ssq
#                 ssq = np.sum(ee**2)
#                 alpha *= 4.0
#                 nit2 += 1
#
#             alpha /= 8.0
#             fakt = 0.5
#             nit2 = 0
#
#             while ssq <= ssqo and nit2 < maxit2:
#                 fakt *= 2.0
#                 r1[:3] += vt[:3] * fakt
#                 d1 += vt[3] * fakt
#
#                 nr1 = 2.0 * (r1 - m) / (a**2)
#                 ee[0] = np.dot(vs, r1) + vs4
#                 ee[1] = -1.0 + np.sum(((r1 - m) / a)**2)
#                 ee[2] = np.dot(nr1, r1) + d1
#                 ee[3] = np.dot(nr1, p1) + d1
#
#                 ssqo = ssq
#                 ssq = np.sum(ee**2)
#                 nit2 += 1
#
#             r1[:3] -= vt[:3] * fakt
#             d1 -= vt[3] * fakt
#
#             nr1 = 2.0 * (r1 - m) / (a**2)
#             ee[0] = np.dot(vs, r1) + vs4
#             ee[1] = -1.0 + np.sum(((r1 - m) / a)**2)
#             ee[2] = np.dot(nr1, r1) + d1
#             ee[3] = np.dot(nr1, p1) + d1
#
#             ssq = np.sum(ee**2)
#             ssqo = ssq
#
#         return 1
#
#     def calc_distance_on_ellipsoid(self, r1, r2, m, a, vs, vs4, far_side_wrap, result):
#         """Calculate distance on ellipsoid surface. Direct translation from OpenSim."""
#         dr = r1 - r2
#         length = np.linalg.norm(dr) / result.factor
#
#         desired_seg_length = 0.001
#
#         if length < desired_seg_length:
#             result.wrap_pts = [r1.copy(), r2.copy()]
#             result.wrap_path_length = length * result.factor
#             return
#
#         num_path_segments = int(length / desired_seg_length)
#         if num_path_segments <= 0:
#             result.wrap_path_length = length
#             return
#         if num_path_segments > 499:
#             num_path_segments = 499
#
#         num_interior_pts = num_path_segments - 1
#
#         # Find major axis
#         imax = np.argmax(np.abs(vs))
#         u = np.zeros(3)
#         u[imax] = 1.0
#
#         mu = (-(np.dot(vs, m)) - vs4) / np.dot(vs, u)
#         a0 = m + mu * u
#
#         ar1 = normalize_or_zero(r1 - a0)
#         ar2 = normalize_or_zero(r2 - a0)
#
#         phi0 = np.arccos(np.dot(ar1, ar2))
#
#         if far_side_wrap:
#             dphi = -(2 * np.pi - phi0) / num_path_segments
#         else:
#             dphi = phi0 / num_path_segments
#
#         vsz = normalize_or_zero(np.cross(ar1, ar2))
#         vsy = np.cross(vsz, ar1)
#
#         r0 = np.column_stack([ar1, vsy, vsz])
#         ux = np.array([1.0, 0.0, 0.0])
#
#         s = []
#         for i in range(num_interior_pts):
#             phi = (i + 1) * dphi
#             rphi = np.array([[np.cos(phi), -np.sin(phi), 0],
#                             [np.sin(phi), np.cos(phi), 0],
#                             [0, 0, 1]])
#
#             t = rphi @ ux
#             r = r0 @ t
#
#             f1 = r / a
#             f2 = (a0 - m) / a
#             aa = np.dot(f1, f1)
#             bb = 2.0 * np.dot(f1, f2)
#             cc = np.dot(f2, f2) - 1.0
#             mu3 = (-bb + np.sqrt(bb**2 - 4.0 * aa * cc)) / (2.0 * aa)
#
#             s.append(a0 + mu3 * r)
#
#         result.wrap_pts = [r1.copy()] + s + [r2.copy()]
#
#         result.wrap_path_length = 0.0
#         for i in range(num_path_segments):
#             dv = result.wrap_pts[i+1] - result.wrap_pts[i]
#             result.wrap_path_length += np.linalg.norm(dv)
#
#     def find_closest_point(self, a, b, c, u, v, w, special_case_axis=-1):
#         """
#         Find closest point on ellipsoid to a point in space.
#         Direct translation from OpenSim (Dave Eberly's algorithm).
#         """
#         # Handle special cases near coordinate planes
#         if special_case_axis < 0:
#             uvw = np.array([u, v, w])
#             min_ellipse_radii_sum = np.inf
#
#             for i in range(3):
#                 if EQUAL_WITHIN_ERROR(0.0, uvw[i]):
#                     ellipse_radii_sum = np.sum(self.dimensions) - self.dimensions[i]
#                     if min_ellipse_radii_sum > ellipse_radii_sum:
#                         special_case_axis = i
#                         min_ellipse_radii_sum = ellipse_radii_sum
#
#         if special_case_axis == 0:
#             x = u
#             y, z, _ = self.closest_point_to_ellipse(b, c, v, w)
#             return np.array([x, y, z])
#         elif special_case_axis == 1:
#             y = v
#             z, x, _ = self.closest_point_to_ellipse(c, a, w, u)
#             return np.array([x, y, z])
#         elif special_case_axis == 2:
#             z = w
#             x, y, _ = self.closest_point_to_ellipse(a, b, u, v)
#             return np.array([x, y, z])
#
#         # General case
#         a2, b2, c2 = a*a, b*b, c*c
#         u2, v2, w2 = u*u, v*v, w*w
#         a2u2, b2v2, c2w2 = a2*u2, b2*v2, c2*w2
#
#         # Initial guess
#         if (u/a)**2 + (v/b)**2 + (w/c)**2 < 1.0:
#             t = 0.0
#         else:
#             max_dim = max(a, b, c)
#             t = max_dim * np.sqrt(u2 + v2 + w2)
#
#         # Newton iteration
#         for _ in range(64):
#             P = t + a2
#             Q = t + b2
#             R = t + c2
#             P2, Q2, R2 = P*P, Q*Q, R*R
#
#             f = P2*Q2*R2 - a2u2*Q2*R2 - b2v2*P2*R2 - c2w2*P2*Q2
#
#             if abs(f) < 1e-09:
#                 x = a2 * u / P
#                 y = b2 * v / Q
#                 z = c2 * w / R
#                 return np.array([x, y, z])
#
#             PQ, PR, QR = P*Q, P*R, Q*R
#             PQR = P*Q*R
#             fp = 2.0 * (PQR*(QR+PR+PQ) - a2u2*QR*(Q+R) - b2v2*PR*(P+R) - c2w2*PQ*(P+Q))
#
#             t -= f / fp
#
#         # Fallback
#         x = a2 * u / (t + a2)
#         y = b2 * v / (t + b2)
#         z = c2 * w / (t + c2)
#         return np.array([x, y, z])
#
#     def closest_point_to_ellipse(self, a, b, u, v):
#         """
#         Find closest point on 2D ellipse to a point.
#         Direct translation from OpenSim (Dave Eberly's algorithm).
#         """
#         a2, b2 = a*a, b*b
#         u2, v2 = u*u, v*v
#         a2u2, b2v2 = a2*u2, b2*v2
#
#         near_x_origin = EQUAL_WITHIN_ERROR(0.0, u)
#         near_y_origin = EQUAL_WITHIN_ERROR(0.0, v)
#
#         # Handle points near axes
#         if near_x_origin and near_y_origin:
#             if a < b:
#                 x = -a if u < 0.0 else a
#                 y = v
#                 return x, y, a
#             else:
#                 x = u
#                 y = -b if v < 0.0 else b
#                 return x, y, b
#
#         if near_x_origin:
#             if a >= b or abs(v) >= b - a2/b:
#                 x = u
#                 y = b if v >= 0 else -b
#                 dy = y - v
#                 return x, y, abs(dy)
#             else:
#                 y = b2 * v / (b2 - a2)
#                 dy = y - v
#                 ydb = y / b
#                 x = a * np.sqrt(abs(1 - ydb*ydb))
#                 return x, y, np.sqrt(x*x + dy*dy)
#
#         if near_y_origin:
#             if b >= a or abs(u) >= a - b2/a:
#                 x = a if u >= 0 else -a
#                 dx = x - u
#                 y = v
#                 return x, y, abs(dx)
#             else:
#                 x = a2 * u / (a2 - b2)
#                 dx = x - u
#                 xda = x / a
#                 y = b * np.sqrt(abs(1 - xda*xda))
#                 return x, y, np.sqrt(dx*dx + y*y)
#
#         # General case - Newton iteration
#         if (u/a)**2 + (v/b)**2 < 1.0:
#             t = 0.0
#         else:
#             max_dim = max(a, b)
#             t = max_dim * np.sqrt(u2 + v2)
#
#         for _ in range(64):
#             P = t + a2
#             Q = t + b2
#             P2, Q2 = P*P, Q*Q
#             f = P2*Q2 - a2u2*Q2 - b2v2*P2
#
#             if abs(f) < 1e-09:
#                 break
#
#             fp = 2.0 * (P*Q*(P+Q) - a2u2*Q