"""FastAPI factory for the local FairFlex Demonstrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .repository import RunNotFoundError
from .schemas import ComparisonRequest, RunAccepted, RunCreateRequest
from .service import APP_VERSION, DemonstratorService


def _raise_not_found(message: str) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "not_found", "message": message})


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Create the credential-free, localhost-oriented demonstrator service."""

    resolved_data_dir = data_dir or Path("data") / "app"
    service = DemonstratorService(resolved_data_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        service.shutdown()

    app = FastAPI(
        title="FairFlex Demonstrator API",
        version=APP_VERSION,
        description="Auditable offline controlled EV-charging teaching simulations. It never controls a physical charger.",
        lifespan=lifespan,
    )
    app.state.fairflex_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def local_security_headers(_request: Request, call_next):
        """Small safe defaults for the loopback-only demonstrator."""

        response = await call_next(_request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
        )
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        return service.catalog.list_public()

    @app.get("/api/scenarios/{scenario_id}")
    def scenario_detail(scenario_id: str) -> dict[str, Any]:
        try:
            return service.catalog.snapshot(scenario_id)
        except KeyError:
            _raise_not_found("Unknown approved scenario")

    @app.post("/api/runs", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
    def create_run(request: RunCreateRequest) -> dict[str, str]:
        try:
            return service.start_run(request.scenario_id, request.include_v3_shadow)
        except KeyError:
            _raise_not_found("Unknown approved scenario")

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return service.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return service.get_run(run_id)
        except RunNotFoundError:
            _raise_not_found("Run not found")

    @app.post("/api/runs/{run_id}/comparisons", status_code=status.HTTP_202_ACCEPTED)
    def compare_v3(run_id: str, request: ComparisonRequest) -> dict[str, str]:
        if request.mode != "v3_shadow":
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported comparison mode")
        try:
            return service.start_v3_shadow(run_id)
        except RunNotFoundError:
            _raise_not_found("Run not found")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "run_not_ready", "message": str(exc)}) from exc

    @app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
    def report(run_id: str) -> Response:
        try:
            content = service.report(run_id)
        except RunNotFoundError:
            _raise_not_found("Run not found")
        return Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="fairflex-audit-{run_id}.md"'},
        )

    @app.get("/api/evidence")
    def evidence() -> dict[str, Any]:
        return service.evidence()

    @app.get("/api/evidence/report", response_class=PlainTextResponse)
    def evidence_report() -> Response:
        return Response(
            content=service.evidence_report(),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="fairflex-evidence-ledger.md"'},
        )

    # The production launcher builds the Vite client once, then this local
    # FastAPI process serves the resulting static dashboard on the same origin
    # as the API.  Development may still use Vite's loopback proxy on :5173.
    dist_dir = Path(__file__).resolve().parents[3] / "web" / "dist"
    index_file = dist_dir / "index.html"
    if index_file.is_file():
        assets_dir = dist_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="built-assets")

        @app.get("/", include_in_schema=False)
        def built_dashboard() -> FileResponse:
            return FileResponse(index_file, headers={"Cache-Control": "no-store"})

    return app
