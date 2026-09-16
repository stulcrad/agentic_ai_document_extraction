import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.model.loader import ensure_model_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pull and load the model before the first request arrives."""
    logging.basicConfig(level=logging.INFO)
    ensure_model_ready()
    yield


app = FastAPI(
    title="Contract Field Extractor",
    description="Extracts seven contract fields as JSON using a local LLM with schema-constrained decoding.",
    lifespan=lifespan,
)
app.include_router(router)
