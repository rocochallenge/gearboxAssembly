"""Compatibility import for the shared R1Pro / R1 Lite solver profile."""

from .physics_profiles import use_fast_physics as use_fast_r1pro_physics

__all__ = ["use_fast_r1pro_physics"]
