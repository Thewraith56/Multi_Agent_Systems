"""solver/vehicles — concrete BaseVehicle implementations.

Currently implemented:
    UnicycleVehicle   — dynamic (acceleration-controlled) unicycle, n=5, m=2

Planned (see base.py module docstring), not yet implemented in this package:
    Drone             — 2-D double integrator, n=4, m=2
    Bicycle           — kinematic bicycle, n=5, m=2
"""

from .base import BaseVehicle
from .unicycle import UnicycleVehicle

__all__ = ["BaseVehicle", "UnicycleVehicle"]
