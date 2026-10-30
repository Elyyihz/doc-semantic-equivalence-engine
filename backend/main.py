import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.presentation.api.v1.router import router as api_v1_router
from app.infrastructure.config.settings import settings
from app.infrastructure.config.telemetry import setup_telemetry

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("semantic_engine.main")

app = FastAPI(
    title="Motor de Equivalência Semântica de Documentos (Graph-Based)",
    description="Motor RAG e LangGraph assíncrono para validação de equivalência documental e conformidade semântica com PaddleOCR, Qdrant e vLLM.",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry Tracing
setup_telemetry(app)

# Presentation Routers
app.include_router(api_v1_router)


@app.get("/")
def root():
    """Root redirect and basic descriptor."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
