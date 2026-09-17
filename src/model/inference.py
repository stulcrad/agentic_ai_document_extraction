import json
import time
import unicodedata

from src.api.schemas import ExtractResponse, FieldResult, FieldStatus
from src.config import get_settings
from src.model.fields import ContractExtraction
from src.model.loader import get_client

# Marks where the middle of a long contract was cut out.
TRUNCATION_MARKER = "\n[... middle of document omitted ...]\n"

# Read once at import; get_settings() is cached, so this is the same instance everywhere.
settings = get_settings()

def normalise(text: str) -> str:
    """Return text in a form suitable for substring comparison."""
    
    # Step 1: Unicode NFKC normalisation
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Collapse whitespace
    text = " ".join(text.split())

    # Step 3: Drop quote characters; the models often swaps " for ' or typographic quotes
    text = text.translate(str.maketrans("", "", "\"'“”‘’"))

    # Step 4: Casefold
    text = text.casefold()

    return text


def truncate_document(text: str, max_chars: int) -> tuple[str, bool]:
    """Shorten text to max_chars by keeping its head and tail; also report whether anything was cut."""
    if len(text) <= max_chars:
        return text, False
    else:
        # Calculate the lengths of the head and tail segments
        # Keep 2/3 of the head and 1/3 of the tail, minus the length of the truncation marker
        marker_length = len(TRUNCATION_MARKER)
        head_length = (2 * max_chars - marker_length) // 3
        tail_length = max_chars - head_length - marker_length

        # Extract the head and tail segments
        head = text[:head_length]
        tail = text[-tail_length:]

        # Combine them with the truncation marker
        truncated_text = head + TRUNCATION_MARKER + tail

        return truncated_text, True


def build_prompt(document: str) -> str:
    """Build the single user message: rules, the JSON schema, then the document."""
    rules = (
        "Extract the fields from the contract below. "
        "Use only what is written in the document. If a field is not stated, set it to null. "
        "Never infer, guess or calculate a value. "
        "For each field, give the shortest verbatim quote from the document that supports its value. "
        "Return JSON that matches this schema:"
    )
    schema = ContractExtraction.model_json_schema()
    prompt = f"{rules}\n{json.dumps(schema)}\n\n=== CONTRACT ===\n{document}\n=== END CONTRACT ==="
    return prompt


def verify_field(name: str, value: object, evidence: str | None, document: str) -> tuple[FieldStatus, str | None]:
    """
    Decide one field's status.

    - `name` is the field name
    - `value` and `evidence` are the extracted values
    - `document` is the normalised text that was sent to the model.

    Return a tuple of (status, reason_tag). The reason tag is None when the field is extracted, or a short
    string explaining why it is unverified.
    1. No value -> not found. The model followed the "return null" rule; there is nothing to verify.
    2. A value but no quote -> unverified: an unsupported claim.
    3. The normalised quote is not a substring of `document` -> unverified. The model invented or 
       paraphrased it. This check is what makes the brief's "flag what you are unsure about" rule real.
    4. Dates: the value's year must appear in the quote, if there is a year in the evidence.
    5. Governing law and renewal term: the value must appear in the quote.
    6. Parties: every party name must appear in `document`, otherwise unverified.
    7. Anything that survives -> extracted.
    """
    if value is None:
        return FieldStatus.NOT_FOUND, None
    if evidence is None:
        return FieldStatus.UNVERIFIED, "no_quote"
    if normalise(evidence) not in document:
        return FieldStatus.UNVERIFIED, "quote_not_in_doc"
    if name in ("governing_law", "renewal_term") and normalise(value) not in normalise(evidence):
        return FieldStatus.UNVERIFIED, "value_not_in_quote"
    if name in ["agreement_date", "effective_date", "expiration_date"]:
        year = str(value.year)
        if year not in evidence and f"/{year[2:]}" not in evidence:
            return FieldStatus.UNVERIFIED, "year_not_in_quote"
    if name == "parties":
        for party in value:
            if normalise(party.name) not in document:
                return FieldStatus.UNVERIFIED, "party_name_not_in_doc"
    return FieldStatus.FOUND, None


def extract_fields(text: str) -> ExtractResponse:
    """Run one extraction end to end: truncate, prompt, generate, validate, verify."""
    # Build
    document, truncated = truncate_document(text, settings.max_doc_chars)
    prompt = build_prompt(document)

    # Generate
    start_time = time.perf_counter()
    response = get_client().chat(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
        format=ContractExtraction.model_json_schema(),
        options={"temperature": settings.temperature, "num_ctx": settings.num_ctx},
        keep_alive=settings.keep_alive,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Validate and verify
    validated = ContractExtraction.model_validate_json(response.message.content)
    normalised_doc = normalise(document)
    results = {}
    for field_name in ContractExtraction.model_fields:
        field_value = getattr(validated, field_name)
        status, reason_tag = verify_field(
            field_name, field_value.value, field_value.evidence, normalised_doc
        )
        results[field_name] = FieldResult(
            value=field_value.value,
            quote=field_value.evidence,
            status=status,
            reason=reason_tag,
        )
    
    return ExtractResponse(
        results=results,
        model_name=settings.model_name,
        truncated=truncated,
        prompt_tokens=response.prompt_eval_count,
        elapsed_ms=elapsed_ms,
    )
