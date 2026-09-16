import logging
from functools import lru_cache

from ollama import Client

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_client() -> Client:
    """Return a cached Ollama client for the configured host."""
    settings = get_settings()
    return Client(host=settings.ollama_host, timeout=settings.request_timeout_s)


def is_model_available() -> bool:
    """Return True if Ollama is reachable and the configured model is pulled."""
    try:
        names = [m.model for m in get_client().list().models]
        return get_settings().model_name in names
    except ConnectionError:
        return False


def ensure_model_ready() -> None:
    """Pull the configured model if it is missing, then load it into memory."""
    settings = get_settings()
    if not is_model_available():
        logger.info("Pulling model %s ...", settings.model_name)
        last_percent = -1
        for progress in get_client().pull(settings.model_name, stream=True):
            if progress.total:
                if progress.completed is None:
                    logger.info("Pulling model %s: %s", settings.model_name, progress.status)
                else:
                    percent = int(progress.completed / progress.total * 100)
                    if percent % 10 == 0 and percent != last_percent:
                        last_percent = percent
                        logger.info("Pulling model %s: %s (%d%%)", 
                                    settings.model_name, progress.status, percent)
        logger.info("Model %s pulled.", settings.model_name)
    get_client().generate(
        model=settings.model_name,
        keep_alive=settings.keep_alive,
        options={"num_ctx": settings.num_ctx},
    )
