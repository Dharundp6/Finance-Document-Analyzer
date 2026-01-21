"""
Main application entry point for the Financial RAG system.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config.logging_config import logger
from config.settings import settings
from core.exceptions import FinancialRAGException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize components
    from api.routes import retriever

    retriever.build_bm25_index()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Application shutting down")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready Financial RAG System using Google Gemini",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(FinancialRAGException)
async def financial_rag_exception_handler(request, exc: FinancialRAGException):
    """Handle custom exceptions."""
    return JSONResponse(status_code=400, content=exc.to_dict())


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": str(exc) if settings.debug else "An internal error occurred",
        },
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
