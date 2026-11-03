"""nanoBragg.c's lengths of the cell axes, to the bit.

C rebuilds the cell through its reciprocal vectors before it measures it, so a[0] is not quite the
number given: 120 A comes back one ulp away. That is enough to move ceil(3*xtalsize/rps) wherever
the argument is a whole number, so this follows C's order of operations in Python floats, whose
sin, cos and sqrt are C's libm. Nothing here is differentiable; it is only for choosing an integer.
"""

import math
from typing import Optional, Sequence, Tuple

Vector = Tuple[float, float, float]

RTD = 180.0 * 0.318309886183790671537767526745  # 180.0*M_1_PI


def _cross(x: Vector, y: Vector) -> Vector:
    return (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])


def _dot(x: Vector, y: Vector) -> float:
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def _magnitude(v: Vector) -> float:
    return math.sqrt(_dot(v, v))


def _scale(v: Vector, scale: float) -> Vector:
    return (scale * v[0], scale * v[1], scale * v[2])


def _rescale(v: Vector, new_magnitude: float) -> Vector:
    """vector_rescale: the quotient comes first, as in C."""
    old = _magnitude(v)
    if old <= 0.0:
        old = 1.0
    return (new_magnitude / old * v[0], new_magnitude / old * v[1], new_magnitude / old * v[2])


def _rotate(v: Vector, phix: float, phiy: float, phiz: float) -> Vector:
    x, y, z = v
    if phix != 0:
        y, z = (y * math.cos(phix) + z * -math.sin(phix), y * math.sin(phix) + z * math.cos(phix))
    if phiy != 0:
        x, z = (x * math.cos(phiy) + y * 0 + z * math.sin(phiy), x * -math.sin(phiy) + y * 0 + z * math.cos(phiy))
    if phiz != 0:
        x, y = (x * math.cos(phiz) + y * -math.sin(phiz), x * math.sin(phiz) + y * math.cos(phiz))
    return (x, y, z)


def _reciprocal_from_cell(a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
    """nanoBragg.c:1791-1878, for a user cell."""
    aavg = (alpha + beta + gamma) / 2
    skew = math.sin(aavg) * math.sin(aavg - alpha) * math.sin(aavg - beta) * math.sin(aavg - gamma)
    V_cell = 2.0 * a * b * c * math.sqrt(abs(skew))
    if V_cell <= 0.0:
        V_cell = 2.2250738585072014e-308  # DBL_MIN
    V_star = 1.0 / V_cell
    a_star = b * c * math.sin(alpha) * V_star
    b_star = c * a * math.sin(beta) * V_star
    c_star = a * b * math.sin(gamma) * V_star
    sin_gamma_star = c * V_star / a_star / b_star
    cos_alpha_star = (math.cos(beta) * math.cos(gamma) - math.cos(alpha)) / (math.sin(beta) * math.sin(gamma))
    cos_beta_star = (math.cos(gamma) * math.cos(alpha) - math.cos(beta)) / (math.sin(gamma) * math.sin(alpha))
    cos_gamma_star = (math.cos(alpha) * math.cos(beta) - math.cos(gamma)) / (math.sin(alpha) * math.sin(beta))
    sin_gamma_star = min(max(sin_gamma_star, -1.0), 1.0)
    if cos_gamma_star * cos_gamma_star > 1.0:
        cos_gamma_star = 1.0
    if cos_beta_star * cos_beta_star > 1.0:
        cos_beta_star = 1.0
    if cos_alpha_star * cos_alpha_star > 1.0:
        cos_alpha_star = 1.0
    return (
        (a_star, 0.0, 0.0),
        (b_star * cos_gamma_star, b_star * sin_gamma_star, 0.0),
        (c_star * cos_beta_star,
         c_star * (cos_alpha_star - cos_beta_star * cos_gamma_star) / sin_gamma_star,
         c_star * V_cell / (a * b * c * sin_gamma_star)),
    ), V_cell


def c_cell_lengths_m(
    cell: Sequence[float],
    misset_deg: Sequence[float] = (0.0, 0.0, 0.0),
    reciprocal: Optional[Sequence[Vector]] = None,
    user_cell: bool = True,
) -> Vector:
    """a[0], b[0], c[0] as nanoBragg.c:2071 has them, in metres.

    Args:
        cell: a, b, c in A, alpha, beta, gamma in degrees
        misset_deg: the misset angles
        reciprocal: a*, b*, c* of a matrix, in 1/A, in place of the cell's orientation
        user_cell: whether -cell was given; with a matrix it still sets the lengths and volume
            (nanoBragg.c:1935-1942)
    """
    a, b, c = cell[:3]
    V_user = None
    if user_cell:
        stars, V_user = _reciprocal_from_cell(a, b, c, *(angle / RTD for angle in cell[3:]))
    if reciprocal is not None:
        stars = reciprocal
    a_star, b_star, c_star = (_rotate(star, *(angle / RTD for angle in misset_deg)) for star in stars)

    a_star_cross_b_star = _cross(a_star, b_star)
    b_star_cross_c_star = _cross(b_star, c_star)
    c_star_cross_a_star = _cross(c_star, a_star)
    V_star = _dot(a_star, b_star_cross_c_star)
    if V_user is not None:
        b_star_cross_c_star = _rescale(b_star_cross_c_star, a / V_user)
        c_star_cross_a_star = _rescale(c_star_cross_a_star, b / V_user)
        a_star_cross_b_star = _rescale(a_star_cross_b_star, c / V_user)
        V_star = 1.0 / V_user
    V_cell = 1.0 / V_star
    real = (_scale(b_star_cross_c_star, V_cell), _scale(c_star_cross_a_star, V_cell),
            _scale(a_star_cross_b_star, V_cell))
    return tuple(_magnitude(_scale(v, 1e-10)) for v in real)


def c_ceil_to_int(x: float) -> int:
    """(int)ceil(x) on x86, so anything outside int comes back as INT_MIN."""
    if math.isnan(x) or math.isinf(x):
        return -2147483648
    result = math.ceil(x)
    return result if -2147483648 <= result <= 2147483647 else -2147483648
