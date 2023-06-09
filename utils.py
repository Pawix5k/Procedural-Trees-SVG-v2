from __future__ import annotations

import math

import numpy as np


class Angle:
    def __init__(self, value: float):
        self.value = self.normalize_value(value)
    
    def __float__(self) -> float:
        return self.value
    
    def __add__(self, other: Angle) -> Angle:
        return Angle(self.value + other.value)
    
    def normalize_value(self, value: float) -> float:
        value %= 360.0
        if value > 180.0:
            return value - 360.0
        return value


class Vector:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    
    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)
    
    def __radd__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)
    
    def __rsub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other: float) -> Vector:
        return Vector(self.x * other, self.y * other)
    
    def __rmul__(self, other: float) -> Vector:
        return Vector(self.x * other, self.y * other)
    
    def rotate(self, angle: float | Angle) -> Vector:
        x = self.x * math.cos(math.radians(angle)) - self.y * math.sin(math.radians(angle))
        y = self.x * math.sin(math.radians(angle)) + self.y * math.cos(math.radians(angle))
        return Vector(x, y)
    
    def to_str(self) -> str:
        return f"{self.x:.2f} {self.y:.2f}"


def point_to_str(point, sep=" "):
    return sep.join(f"{coord:.2f}" for coord in point)


maple_points = np.array(
    [
        [0.0, 0.0],
        [-0.25, 0.04],
        [-0.11, -0.16],
        [-0.42, -0.25],
        [-0.5, -0.5],
        [-0.3, -0.61],
        [-0.1, -0.46],
        [-0.2, -0.76],
        [0.0, -0.96],
        [0.2, -0.76],
        [0.1, -0.46],
        [0.3, -0.61],
        [0.5, -0.5],
        [0.42, -0.25],
        [0.11, -0.16],
        [0.25, 0.04],
    ]
)

round_points = np.array(
    [
        [0.0, 0.0],
        [-0.191, -0.038],
        [-0.354, -0.146],
        [-0.462, -0.309],
        [-0.5, -0.5],
        [-0.462, -0.691],
        [-0.354, -0.854],
        [-0.191, -0.962],
        [0.0, -1],
        [0.191, -0.962],
        [0.354, -0.854],
        [0.462, -0.691],
        [0.5, -0.5],
        [0.462, -0.309],
        [0.354, -0.146],
        [0.191, -0.038],
    ]
)

normal_points = np.array(
    [
        [0.0, 0.0],
        [-0.23, -0.03],
        [-0.41, -0.13],
        [-0.5, -0.309],
        [-0.46, -0.51],
        [-0.33, -0.61],
        [-0.16, -0.68],
        [-0.03, -0.82],
        [0.0, -1.0],
        [0.03, -0.82],
        [0.16, -0.68],
        [0.33, -0.61],
        [0.46, -0.51],
        [0.5, -0.309],
        [0.41, -0.13],
        [0.23, -0.03],
    ]
)

needle_points = np.array(
    [
        [0.0, 0.0],
        [-0.02, -0.04],
        [-0.07, -0.22],
        [-0.13, -0.4],
        [-0.14, -0.5],
        [-0.13, -0.6],
        [-0.07, -0.78],
        [-0.02, -0.88],
        [0.0, -1.0],
        [0.02, -0.88],
        [0.07, -0.78],
        [0.13, -0.6],
        [0.14, -0.5],
        [0.13, -0.4],
        [0.07, -0.22],
        [0.02, -0.04],
    ]
)

leaf_points = {
    "maple": maple_points,
    "round": round_points,
    "normal": normal_points,
    "needle": needle_points,
}

intervals = (
    ("maple", 0.0),
    ("round", 0.33),
    ("normal", 0.66),
    ("needle", 1.0),
)


def get_leaf_interpolation_proportions(x):
    closest = []
    for i, interval in enumerate(intervals):
        if x <= interval[1]:
            if i > 0:
                closest.append(intervals[i - 1])
            closest.append(interval)
            break
    if len(closest) == 1:
        return ((closest[0][0], 1.0),)
    first, second = closest
    distance = second[1] - first[1]
    second_prop = (x - first[1]) / distance
    first_prop = 1.0 - second_prop
    return (first[0], first_prop), (second[0], second_prop)


def get_raw_leaf_points(x):
    proportions = get_leaf_interpolation_proportions(x)
    if len(proportions) == 1:
        points = leaf_points[proportions[0][0]] * 1.0
    else:
        first_name, first_proportion = proportions[0]
        second_name, second_proportion = proportions[1]
        points = (
            first_proportion * leaf_points[first_name]
            + second_proportion * leaf_points[second_name]
        )
    
    return points


def get_rotation_matrix(angle):
    return np.array(
        [
            [
                np.cos(np.radians(angle.value)),
                np.sin(np.radians(angle.value)),
            ],
            [
                -np.sin(np.radians(angle.value)),
                np.cos(np.radians(angle.value)),
            ],
        ]
    )


def get_point_at_quad_bez(p1, c, p2, t):
    p3 = p1 + t * (c - p1)
    p4 = c + t * (p2 - c)
    result = p3 + t * (p4 - p3)
    return result
