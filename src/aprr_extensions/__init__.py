"""
APRR Extensions: CROW and OctoRoute routing variants.

Modules
-------
crow_router      : Chain-of-Reasoning Over Workload Router
octoroute_router : Distributed Parallel-Dispatch Routing
"""
from .crow_router import CROWRouter
from .octoroute_router import DispatchArm, OctoRouteRouter

__all__ = ["CROWRouter", "DispatchArm", "OctoRouteRouter"]
