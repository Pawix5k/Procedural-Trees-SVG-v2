from __future__ import annotations

import math


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


def point_to_str(point):
    return " ".join(f"{coord:.2f}" for coord in point)
