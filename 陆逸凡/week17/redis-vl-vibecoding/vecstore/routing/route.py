"""Route definition data structures for the semantic router.

A :class:`Route` defines an intent category with example utterances.
:class:`RouteMatch` is the result of classifying an input against the routes.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class Route:
    """A single route / intent for the :class:`SemanticRouter`.

    Attributes:
        name: Unique identifier for this route (e.g. ``"greeting"``, ``"weather"``).
        examples: Example utterances that belong to this route.  These are
            embedded at initialization time and form the "prototypes" that
            incoming text is compared against.
        description: Human-readable description of the intent.
        handler: Optional async callable invoked when this route is matched.
            The signature is ``handler(input_text, **metadata)``.
        metadata: Arbitrary extra data attached to the route.
    """

    name: str
    """Unique route name."""

    examples: List[str]
    """Example utterances for this route."""

    description: str = ""
    """Human-readable description."""

    handler: Optional[Callable[..., Awaitable[Any]]] = None
    """Optional handler callable."""

    metadata: Optional[dict] = None
    """Extra metadata."""


@dataclass
class RouteMatch:
    """Result of routing an input text.

    Attributes:
        route: The matched :class:`Route`.
        input_text: The original input that was classified.
        distance: Cosine distance of the best-matching example [0, 2].
        confidence: Normalized confidence ``1.0 - (distance / 2.0)`` in [0, 1].
    """

    route: Route
    """The matched route."""

    input_text: str
    """The original input text."""

    distance: float
    """Cosine distance of the best-matching example."""

    confidence: float
    """Confidence score in [0, 1] (``1 - distance/2``)."""
