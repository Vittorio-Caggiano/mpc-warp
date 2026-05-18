from __future__ import annotations


def quadratic_cost(state: list[float], action: list[float], goal: list[float], q: float = 1.0, r: float = 0.01) -> float:
    se = sum((s - g) * (s - g) for s, g in zip(state, goal))
    au = sum(u * u for u in action)
    return float(q * se + r * au)
