from fastapi import APIRouter, HTTPException, Response
from ollama import ResponseError
from pydantic import ValidationError

from src.api.schemas import (
    DatasetItem,
    DemoResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    SchemaResponse,
)
from src.config import get_settings
from src.dataset import load_test_set
from src.model.fields import ContractExtraction
from src.model.inference import extract_fields
from src.model.loader import is_model_available

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Report whether the service can currently serve extractions."""

    if is_model_available():
        return HealthResponse(
            status="ready",
            model_pulled=True,
            model_name=settings.model_name,
        )
    else:
        response.status_code = 503
        return HealthResponse(
            status="starting",
            model_pulled=False,
            model_name=settings.model_name,
        )


@router.get("/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    """Return the extraction schema and one description per field."""
    schema = ContractExtraction.model_json_schema()
    descriptions = {name: prop.get("description", "") for name, prop in schema.get("properties", {}).items()}
    return SchemaResponse(body=schema, descriptions=descriptions)


@router.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest) -> ExtractResponse:
    """Extract the seven contract fields from one document."""
    try:
        return extract_fields(body.document)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail="Ollama unreachable") from e
    except (ResponseError, ValidationError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e



@router.get("/dataset", response_model=list[DatasetItem])
def list_dataset() -> list[DatasetItem]:
    """List the test documents bundled with the service."""
    return [
        DatasetItem(
            doc_id=doc_id,
            doc_title=doc.get("title", doc_id),
            doc_length=len(doc["doc_text"]),
            note=doc.get("note"),
        )
        for doc_id, doc in load_test_set().items()
    ]



@router.get("/dataset/{doc_id}/demo", response_model=DemoResponse)
def demo(doc_id: str) -> DemoResponse:
    """Extract from one test document and show the result next to the expected answers."""
    doc = load_test_set().get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
    extraction = extract_fields(doc["doc_text"])
    return DemoResponse(doc_id=doc_id, extraction=extraction, gold_labels=doc.get("gold", {}))
