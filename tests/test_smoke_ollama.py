import json

import pytest

from src.config import get_settings
from src.model.fields import ContractExtraction
from src.model.loader import ensure_model_ready, get_client

pytestmark = pytest.mark.slow

SNIPPET = (
    "DISTRIBUTOR AGREEMENT\n"
    'This Distributor Agreement (the "Agreement") is made and entered into as of the 8th day of May, 2014, '
    'by and between Birch First Global Investments Inc., a Nevada corporation ("Company"), and '
    'Mount Kowledge Holdings Inc. ("Distributor").\n'
    "[...]\n"
    "12. Governing Law. This Agreement shall be governed by and construed under "
    "the laws of the State of Nevada."
)


@pytest.fixture(scope="module")
def raw_response():
    """One structured chat call on SNIPPET, shared by all tests in this module."""
    ensure_model_ready()
    settings = get_settings()
    schema = ContractExtraction.model_json_schema()
    prompt = (
        "Extract the fields from the following contract snippet. "
        "If a field is not stated, set it to null. "
        "For each field, provide the shortest verbatim quote from the document that supports its value. "
        "Return the output in JSON format that matches the provided schema.\n\n"
        + json.dumps(schema)
        + "\n\n"
        + SNIPPET
    )
    response = get_client().chat(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": settings.temperature, "num_ctx": settings.num_ctx},
        keep_alive=settings.keep_alive,
    )
    print(response.prompt_eval_count, response.eval_count, response.total_duration / 1e9)
    print(response.message.content)
    return response


def test_output_parses_into_schema(raw_response):
    """Model output validates against ContractExtraction."""
    ContractExtraction.model_validate_json(raw_response.message.content)



def test_present_field_filled_absent_field_null(raw_response):
    """A field stated in SNIPPET is extracted; a field not stated is null."""
    contract_extraction = ContractExtraction.model_validate_json(raw_response.message.content)
    governing_law = contract_extraction.governing_law.value
    assert governing_law is not None and "Nevada" in governing_law
    assert contract_extraction.expiration_date.value is None


def test_evidence_generated_before_value(raw_response):
    """
    Inside each field object the raw JSON has "evidence" before "value".
    Ollama's JSON streaming sorts the keys alphabetically, so this should be
    true for all fields if the model is generating the JSON in the correct order.
    """
    raw = raw_response.message.content
    raw_json = json.loads(raw)
    for field_name in [
        "contract_type",
        "parties",
        "agreement_date",
        "effective_date",
        "expiration_date",
        "renewal_term",
        "governing_law",
    ]:
        field_obj = raw_json[field_name]
        keys = list(field_obj.keys())
        assert keys.index("evidence") < keys.index("value")
