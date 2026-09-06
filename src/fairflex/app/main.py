"""Local entry point: `python -m fairflex.app.main`."""

from __future__ import annotations

import uvicorn

from .api import create_app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, reload=False)
