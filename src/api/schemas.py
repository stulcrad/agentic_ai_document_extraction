from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FieldStatus(StrEnum):
    """Verdict the service assigns to one extracted field."""
    FOUND = "found_and_verified"
    NOT_FOUND = "not_found"
    UNVERIFIED = "unverified"


class FieldResult(BaseModel):
    """One field as the API returns it: the value, its quote, and our verdict."""
    
    value: Any | None = Field(description="The extracted value. Null unless stated explicitly.")
    quote: str | None = Field(description="Verbatim quote from the document that supports the value.")
    status: FieldStatus = Field(description="Verdict the service assigns to this field.")
    reason: str | None = Field(description="Short reason tag for unverified fields; null otherwise.")

class ExtractRequest(BaseModel):
    """Body of POST /extract."""

    document: str = Field(min_length=1, description="The contract text to extract fields from.")


class ExtractResponse(BaseModel):
    """Result of one extraction, plus what the service did to produce it."""

    results: dict[str, FieldResult] = Field(description="Per-field results, keyed by field name.")
    model_name: str = Field(description="Name of the model that produced this extraction.")
    truncated: bool = Field(description="Whether the document was shortened before sending to the model.")
    prompt_tokens: int | None = Field(description="Prompt token count reported by Ollama.")
    elapsed_ms: float = Field(description="Wall-clock duration of the extraction in milliseconds.")


class HealthResponse(BaseModel):
    """Body of GET /health, returned both when ready and when not."""

    status: str = Field(description="Service status: ready, starting, or error.")
    model_pulled: bool = Field(description="Whether the configured model is pulled and ready to use.")
    model_name: str = Field(description="Name of the model configured for this service.")

class SchemaResponse(BaseModel):
    """Body of GET /schema."""

    body: dict = Field(description="The JSON schema for ContractExtraction.")
    descriptions: dict[str, str] = Field(
        description="Mapping of field name to its description, taken from the schema."
    )


class DatasetItem(BaseModel):
    """One document in the bundled test set."""

    doc_id: str = Field(description="Document id, used in /dataset/{doc_id}/demo.")
    doc_title: str = Field(description="Human-readable title of the document.")
    doc_length: int = Field(description="Length of the document in characters.")
    note: str | None = Field(description="Optional note about why this document is hard to extract.")


class DemoResponse(BaseModel):
    """Body of GET /dataset/{doc_id}/demo: prediction next to the hand-written answers."""

    doc_id: str = Field(description="Document id, used in /dataset/{doc_id}/demo.")
    extraction: ExtractResponse = Field(description="The service's extraction of the document.")
    gold_labels: dict[str, Any | None] = Field(
        description="Mapping of field name to the expected value, taken from the test set."
    )
