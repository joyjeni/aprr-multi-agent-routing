"""
APRR Extensions: CROW and OctoRoute routing variants.

Modules
-------
crow_router      : Chain-of-Reasoning Over Workload Router
octoroute_router : Octopus-Inspired Distributed Routing
"""
from .crow_router import CROWRouter
from .octoroute_router import OctoArm, OctoRouteRouter

__all__ = ["CROWRouter", "OctoArm", "OctoRouteRouter"]
