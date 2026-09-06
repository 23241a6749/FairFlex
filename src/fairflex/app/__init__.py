"""FairFlex Demonstrator application layer.

This package deliberately wraps the research engine without changing it.  It
serves offline, deterministic demonstrations and must never issue a command to
a physical charging station.
"""

from .api import create_app

__all__ = ["create_app"]
