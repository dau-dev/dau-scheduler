"""The two refusals this library makes, named once.

Both are ``ValueError`` subclasses because both mean the same thing: the
caller asked for a split that cannot be planned or cannot be reassembled,
and answering anyway would produce a number nothing computed.
"""

from __future__ import annotations

__all__ = ("PostureError", "SplitError")


class PostureError(ValueError):
    """A posture was requested that cannot be resolved into a share."""


class SplitError(ValueError):
    """A split was asked for that cannot be executed or merged as asked."""
