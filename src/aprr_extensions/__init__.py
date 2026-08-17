"""
APRR Extensions: CTGR and FTDR routing variants.

Modules
-------
ctgr_router : Chain-of-Thought Gated Router
ftdr_router : Functional-Token Dispatch Router

Renamed 2026-08 from ``crow_router``/``octoroute_router`` (classes
``CROWRouter``/``OctoRouteRouter``) to remove biomimicry-derived naming
(CROW, Octo/octopus) per project policy — the extension mechanisms are
now named for what they literally compute: a CoT-gated deliberation
router and a functional-token dispatch router.
"""
from .ctgr_router import CTGRRouter
from .ftdr_router import DispatchArm, FTDRRouter

__all__ = ["CTGRRouter", "DispatchArm", "FTDRRouter"]
