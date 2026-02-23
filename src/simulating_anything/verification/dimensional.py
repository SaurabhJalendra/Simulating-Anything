"""Lightweight dimensional analysis for validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimensions:
    """SI base dimensions: [L, T, M, Theta, N]."""

    length: int = 0
    time: int = 0
    mass: int = 0
    temperature: int = 0
    amount: int = 0

    def __mul__(self, other: Dimensions) -> Dimensions:
        return Dimensions(
            length=self.length + other.length,
            time=self.time + other.time,
            mass=self.mass + other.mass,
            temperature=self.temperature + other.temperature,
            amount=self.amount + other.amount,
        )

    def __truediv__(self, other: Dimensions) -> Dimensions:
        return Dimensions(
            length=self.length - other.length,
            time=self.time - other.time,
            mass=self.mass - other.mass,
            temperature=self.temperature - other.temperature,
            amount=self.amount - other.amount,
        )

    def __pow__(self, n: int) -> Dimensions:
        return Dimensions(
            length=self.length * n,
            time=self.time * n,
            mass=self.mass * n,
            temperature=self.temperature * n,
            amount=self.amount * n,
        )

    @property
    def is_dimensionless(self) -> bool:
        return all(
            v == 0
            for v in (self.length, self.time, self.mass, self.temperature, self.amount)
        )

    def __str__(self) -> str:
        parts = []
        for name, val in [
            ("L", self.length),
            ("T", self.time),
            ("M", self.mass),
            ("Θ", self.temperature),
            ("N", self.amount),
        ]:
            if val == 1:
                parts.append(name)
            elif val != 0:
                parts.append(f"{name}^{val}")
        return " ".join(parts) if parts else "dimensionless"


# Common physical dimensions
DIMENSIONLESS = Dimensions()
LENGTH = Dimensions(length=1)
TIME = Dimensions(time=1)
MASS = Dimensions(mass=1)
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
DIFFUSIVITY = LENGTH**2 / TIME
CONCENTRATION = Dimensions(amount=1) / LENGTH**3
RATE = Dimensions() / TIME


def check_dimensional_consistency(
    lhs_dims: Dimensions, rhs_dims: Dimensions, equation_name: str = ""
) -> tuple[bool, str]:
    """Check that LHS and RHS of an equation have matching dimensions."""
    if lhs_dims == rhs_dims:
        return True, f"{equation_name}: dimensionally consistent ({lhs_dims})"
    return False, (
        f"{equation_name}: dimensional mismatch — LHS={lhs_dims}, RHS={rhs_dims}"
    )
