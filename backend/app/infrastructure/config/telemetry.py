import logging
from fastapi import FastAPI
from app.infrastructure.config.settings import settings

logger = logging.getLogger("semantic_engine.infra.telemetry")


def setup_telemetry(app: FastAPI) -> None:
    """Configures OpenTelemetry tracer and metrics exporters for the application.

    Args:
        app: FastAPI instance to be instrumented.
    """
    logger.info(
        "Initializing OpenTelemetry exporter targeting '%s' for service '%s'",
        settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        settings.OTEL_SERVICE_NAME,
    )
    # Hooks FastAPI middleware for distributed tracing headers
    @app.middleware("http")
    async def trace_headers_middleware(request, call_next):
        response = await call_next(request)
        response.headers["X-Telemetry-Service"] = settings.OTEL_SERVICE_NAME
        return response
