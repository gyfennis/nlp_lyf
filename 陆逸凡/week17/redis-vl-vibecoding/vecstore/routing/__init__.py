"""Routing components — intent recognition and dispatch."""

from vecstore.routing.route import Route, RouteMatch
from vecstore.routing.semantic_router import RouterConfig, SemanticRouter

__all__ = [
    "Route",
    "RouteMatch",
    "RouterConfig",
    "SemanticRouter",
]
