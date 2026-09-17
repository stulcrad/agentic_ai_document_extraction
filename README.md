# Contract Field Extraction with a Local LLM

A FastAPI service that takes the text of a contract and returns seven fields as JSON: contract type, parties,
agreement date, effective date, expiration date, renewal term and governing law. The model is
`qwen3:4b-instruct-2507` (4-bit), served by Ollama. Everything runs locally in Docker, with no API key.

Two layers enforce the brief's rules. Ollama constrains decoding to a JSON schema, so every field is a typed value or
`null` and comes with a verbatim quote from the document. The service then checks each quote against the document
itself and marks the field `found_and_verified`, `not_found` or `unverified`. The model never grades its own answer.

`docker compose up` starts the service. `scripts/evaluate.py` runs it over the test set in `data/test_set/`.

## Quickstart

```bash
docker compose up -d --build --wait
```

The first start downloads about 6 GB once (Ollama image 3.7 GB, model 2.5 GB) and loads the model. `--wait` returns
when both containers are healthy. It runs on CPU: the request below takes about 2 minutes, a full contract 1–5 minutes.

```bash
curl -s -X POST localhost:8000/extract -H 'Content-Type: application/json' \
  -d '{"document": "This Distributor Agreement is made as of May 8, 2014 between Acme Corp. (\"Supplier\") and Beta LLC (\"Distributor\"). This Agreement is governed by the laws of the State of Nevada."}'
```

Each field in the response looks like this:

```json
"governing_law": {"value": "Nevada", "quote": "governed by the laws of the State of Nevada", "status": "found_and_verified", "reason": null}
```

Interactive docs are at http://localhost:8000/docs. Stop with `docker compose down`; add `-v` to also delete the
downloaded model.

### Local development

```bash
conda create -n doc-extract python=3.13 pip -y
conda activate doc-extract
pip install -r requirements.txt -r requirements-dev.txt

ollama serve                          # separate terminal; the app pulls the model on first start
uvicorn src.api.main:app --port 8000  # stop the Docker stack first, it uses port 8000
pytest -m slow                        # smoke test against the running Ollama
python scripts/evaluate.py            # scores the running service on the test set
```

Settings come from environment variables or `.env`; see `.env.example`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Model status. **503 until the model is pulled and loaded**; the Docker healthcheck relies on this |
| GET | `/schema` | The JSON schema the output is constrained to, and one description per field |
| POST | `/extract` | `{"document": "..."}` → value, quote, status and reason per field, plus model, truncation flag, prompt tokens, latency |
| GET | `/dataset` | The bundled test documents, each with a note on why it is hard |
| GET | `/dataset/{doc_id}/demo` | The extraction for one test document next to its gold labels |

Ollama unreachable returns 503. An error from Ollama, or model output that fails validation, returns 502.

## Fields

| Field | Type | Rule given to the model |
|---|---|---|
| `contract_type` | one of 24 contract types, `Other`, or null | null if the document is not a contract |
| `parties` | list of `{name, role}` | all signing parties; role is the contract's defined term |
| `agreement_date` | date | signed or dated "as of" |
| `effective_date` | date | only if stated explicitly |
| `expiration_date` | date | a calendar date only, never computed from a duration |
| `renewal_term` | text | automatic renewal only |
| `governing_law` | text | jurisdiction name only |

The model outputs each field as `{evidence, value}`; the API returns `{value, quote, status, reason}`.

## Data

Documents come from **CUAD v1** (Contract Understanding Atticus Dataset, CC BY 4.0): 510 commercial contracts filed with
the SEC, annotated by lawyers for 41 clause types. Seven of those clause types map onto the fields above. Contract types
come from CUAD's folder names, normalised from 28 inconsistent names to 24 types plus `Other`.

The test set has six documents. One is a baseline; each of the others is hard in a different way:

| Document | Characters | Why it is in the set |
|---|---|---|
| `distributor_baseline` | 6,320 | Every field stated explicitly; minor OCR noise in the date |
| `transportation_three_dates` | 6,088 | Signed 2015, effective 2016, term 2014–2017: the dates must be told apart |
| `sponsorship_two_jurisdictions` | 4,066 | Governing law names California **and** Hong Kong; the field expects one |
| `cobranding_duration_only` | 8,321 | Term given only as "one (1) year"; renewal needs the other party's consent |
| `consulting_truncated` | 20,805 | Over the 16,000-character budget; the governing-law clause is in the removed middle |
| `joint_filing_not_a_commercial_contract` | 645 | An SEC joint-filing statement; the word "Joint" invites `Joint Venture` |

That is 42 fields: 27 stated in the documents and 15 absent, so both recall and the null rule are measured.

**Gold labels follow the service's rules, not CUAD's.** They start from CUAD's answers, checked against the clause
text. Three CUAD answers are overridden:
- two expiration dates that CUAD computed from "one year" after signing, which the never-calculate rule makes null;
- a renewal term that needs the other party's consent, so it is not automatic.

The reasoning for each document is in its `note` in `data/test_set/labels.json`.

To rebuild the test set from source (`data/raw/` is gitignored):

```bash
hf download theatticusproject/cuad --repo-type dataset \
  --include "CUAD_v1/master_clauses.csv" --include "CUAD_v1/full_contract_txt/*" --local-dir data/raw
```

## Method

Per request, in `src/model/inference.py`:

1. **Truncate** documents over 16,000 characters: keep the first two thirds and the last third of the budget, joined by
   a marker. Title, parties and dates sit at the top; governing law and signatures at the bottom.
2. **Prompt** with the rules (use only what is written, null when absent, never infer or calculate, quote verbatim),
   the JSON schema with its field descriptions, and the delimited document.
3. **Generate** with Ollama: `format` set to the schema, `temperature=0`, `num_ctx=8192`.
4. **Validate** the output into the pydantic model `ContractExtraction`.
5. **Verify** each field against the text that was actually sent to the model, after normalising Unicode, whitespace,
   quote characters and case:

| Check, in order | Status | Reason |
|---|---|---|
| value is null | `not_found` | — |
| a value without a quote | `unverified` | `no_quote` |
| quote not in the document | `unverified` | `quote_not_in_doc` |
| governing law or renewal term value not in its quote | `unverified` | `value_not_in_quote` |
| the date's year not in its quote | `unverified` | `year_not_in_quote` |
| a party name not in the document | `unverified` | `party_name_not_in_doc` |
| otherwise | `found_and_verified` | — |

## Design decisions

**Ollama with a 4-bit 4B model, not Hugging Face transformers with XGrammar.** The service has to run on a clean
machine without a GPU. In 4-bit, a 4B model needs about 2.5 GB, roughly what a 0.5B model needs in full precision, and
it is far better at extraction. Ollama still constrains decoding to the schema; what is given up is direct access to
the logits.

**The model quotes before it answers.** Ollama sorts schema keys alphabetically before building the grammar (verified
by testing), so `evidence` is generated before `value` in every field. The model commits to a passage first and extracts
from it. Renaming either key would silently change that order.

**Schema-valid is not correct.** Told to copy a date verbatim while the schema required ISO format, the model produced
`"3333-03-03"`: valid according to the grammar, wrong in fact. The year check exists because of that test.

**The service decides the status, not the model.** A 4B model's self-reported confidence is not calibrated. A quote that
cannot be found in the document is a deterministic, explainable signal, and it catches invented or paraphrased
evidence.

**Truncation is explicit.** Ollama's default context on consumer GPUs is 4,096 tokens, and it silently drops the start
of longer prompts, which is where parties and dates are. `num_ctx` is set on every request, the service cuts long
documents itself, and the response reports it in `truncated`.

**The model is pulled at run time, not built into the image.** The API image stays small, the model is chosen by an
environment variable, and a named volume keeps it across restarts and rebuilds. `/health` returns 503 until the model
is loaded, so the Docker healthcheck and `docker compose up --wait` mean "ready to serve", not "container started".

**Deliberately not agentic.** One constrained call plus deterministic checks is predictable, cheap to evaluate and easy
to explain. Where an agent loop would add value is listed under Future work.

**Deliberately left out:** PDF and scanned input (OCR), authentication, request queueing, GPU deployment and a UI.

## Results

Scored by `scripts/evaluate.py` against the running service. Every field gets exactly one outcome: `correct`,
`correct_null` (absent and returned null), `missed` (stated but returned null), `wrong`, or `hallucinated` (absent but a
value was returned).

- **field_accuracy**: `correct` + `correct_null` over all fields.
- **recall_when_stated**: `correct` over the fields the documents actually state.
- **hallucination_rate_when_absent**: `hallucinated` over the fields the documents do not state. This is the brief's
  "never guess" rule.
- **confidently_wrong**: `wrong` or `hallucinated` fields that still carry `found_and_verified`. These are errors
  presented as fact, the number that matters most for a client.
- **flagged_and_actually_wrong / flagged_but_correct**: whether `unverified` catches real errors or only costs review
  time.

Run against the Docker service on CPU: 6 documents, 42 fields. Every field, with its quote, is in
`results/results.md` and `results/results.json`.

| Metric | Value |
|---|---|
| Field accuracy | **0.81** (34 / 42) |
| Recall when stated | 0.85 (23 / 27) |
| Hallucination rate when absent | 0.27 (4 / 15) |
| Confidently wrong | **7** |
| Flagged `unverified` | 3, all of them correct (false alarms) |

| Field | correct | correct_null | missed | wrong | hallucinated | confidently wrong |
|---|---|---|---|---|---|---|
| contract_type | 4 | 0 | 0 | 2 | 0 | 2 |
| parties | 5 | 0 | 0 | 1 | 0 | 1 |
| agreement_date | 6 | 0 | 0 | 0 | 0 | 0 |
| effective_date | 2 | 3 | 0 | 0 | 1 | 1 |
| expiration_date | 2 | 4 | 0 | 0 | 0 | 0 |
| renewal_term | 0 | 3 | 0 | 0 | 3 | 3 |
| governing_law | 4 | 1 | 1 | 0 | 0 | 0 |

**What the numbers say:**
- **Dates held up.** Agreement date 6/6 and expiration date 6/6, including the two durations CUAD had turned into dates,
  which the service correctly left null.
- **Renewal term is the weakest field.** All three documents without automatic renewal still got a value: a
  termination condition, a renewal that needs the other party's consent, and the contract's duration.
- **Verification flagged none of the 7 wrong or invented values.** Each came with a real quote from the document, and
  the checks prove that a quote exists, not that it supports the value. The 3 flagged fields were all correct and were
  flagged because the model paraphrased: "California and Hong Kong" for "the State of California and the laws of Hong
  Kong", and party quotes shortened with "...".
- **The one miss is the designed one.** Florida sat in the truncated middle of `consulting_truncated`.
- **Contract type:** `Joint Venture` for the joint-filing statement, and `Service` for the consulting agreement. The
  second is defensible, which shows that `Other` in the gold is a labelling choice.

**One failure explained: `distributor_baseline`, effective date.** The contract states no effective date. The model
returned `2021-03-18` with the quote "This Agreement shall be in effect until March 18. 2021": the expiration date,
picked because the sentence says "in effect". The quote exists in the document and contains the year, so both checks
passed and the field came back `found_and_verified`. That is exactly the kind of error the current verification cannot
see, which is why value-level checks come first under Future work.

## Performance

- **Prompt size.** The JSON schema with its descriptions accounts for about 1,100 prompt tokens per request.
- **Latency.** 66–292 seconds per document on CPU in Docker, for prompts of 1.4k–4.4k tokens. The shortest document
  took 66 s on CPU and about 40 s locally with the model split 40/60 between CPU and a 4 GB GTX 1650.
- **Cost.** No per-token cost; everything runs locally.

## Future work

What I would fix first if this had to go in front of a client:

**Check that the value follows from the quote.** Verification proves only that the quote exists, and in the evaluation
it flagged none of the 7 wrong values. Cheap per-field rules would have caught most of them: a renewal quote must
mention renewal, an effective-date quote must say "effective", a contract type must appear in the document's title.
For the rest, a stronger model as judge.

**Chunking instead of truncation.** Split long contracts into overlapping sections, extract per section and merge, so a
clause in the middle, like Florida in `consulting_truncated`, is not lost.

**Evidence constrained to the document.** With a Hugging Face or vLLM backend, a custom logits processor can restrict
the quote to substrings of the input using a token trie, the approach from my master's thesis on span classification
with decoder-only models. Invented quotes then become impossible by construction.

**An agent loop for unverified fields.** Re-ask only the flagged fields, with the relevant section retrieved, instead
of returning them unverified.

**Confidence from log-probabilities.** Ollama returns them. Calibrated on a labelled set, they would rank fields for
human review.

**Unit tests and CI.** The smoke test needs a running model. `verify_field`, `truncate_document` and the evaluation's
`classify` are pure functions and should be tested without one.

**Also:** governing law as a list, a larger test set, PDF input, and GPU serving with vLLM (whose structured-output
backend is XGrammar).

## Final note on time management

Built in about three hours, while on vacation. I prioritised a working service that follows the brief's three rules, a
test set with deliberately hard cases, and an honest evaluation. The omissions and next steps above follow from that
choice.

## Walkthrough

1. `docker compose ps`: both services `(healthy)`.
2. http://localhost:8000/docs → `GET /schema`: the fields and their rules.
3. `GET /dataset` → `GET /dataset/joint_filing_not_a_commercial_contract/demo`: a wrong contract type marked as verified.
4. `GET /dataset/consulting_truncated/demo`: `truncated: true`, governing law missed.
5. Code: `verify_field` in `src/model/inference.py`, the `evidence`/`value` pairs in `src/model/fields.py`, and the
   healthchecks in `docker-compose.yml`.

Each demo call runs a full extraction, so on CPU expect a few minutes per document.

## Project structure

```
.
├── src/
│   ├── config.py               Settings from environment variables / .env
│   ├── dataset.py              Loads the bundled test set
│   ├── api/
│   │   ├── main.py             FastAPI app; pulls and loads the model at start-up
│   │   ├── routes.py           The five endpoints
│   │   └── schemas.py          Request/response models and FieldStatus
│   └── model/
│       ├── fields.py           Extraction schema the LLM output is constrained to
│       ├── loader.py           Ollama client, model pull and warm-up
│       └── inference.py        Truncate, prompt, generate, validate, verify
├── scripts/evaluate.py         Runs the service over the test set, writes results/
├── tests/test_smoke_ollama.py  End-to-end check against a running Ollama
├── data/test_set/              6 documents + labels.json (gold labels and notes)
├── results/                    results.json and results.md from the last evaluation
├── Dockerfile                  API image
├── docker-compose.yml          api + ollama services, model volume, healthchecks
├── requirements.txt            Pinned runtime dependencies, installed in the image
├── requirements-dev.txt        pytest, httpx, ruff, huggingface_hub
├── pyproject.toml              ruff and pytest configuration
└── .env.example                Settings template
```

### Code map

**`src/config.py`**

| Name | What it does |
|---|---|
| `Settings` | `ollama_host`, `model_name`, `num_ctx`, `temperature`, `max_doc_chars`, `keep_alive`, `request_timeout_s`, `test_set_dir` |
| `get_settings()` | Cached `Settings` instance |

**`src/model/fields.py`**

| Name | What it does |
|---|---|
| `ContractType` | 24 contract types from CUAD, plus `Other` |
| `Party` | `name` and `role` |
| `DateField`, `TextField`, `ContractTypeField`, `PartiesField` | `evidence` plus a typed `value` for one field |
| `ContractExtraction` | The seven fields; its JSON schema goes into the prompt and into Ollama's `format` |

**`src/model/loader.py`**

| Name | What it does |
|---|---|
| `get_client()` | Cached Ollama client for the configured host |
| `is_model_available()` | Ollama reachable and model pulled; never raises |
| `ensure_model_ready()` | Pulls the model if missing, logging progress, then loads it with the request `num_ctx` |

**`src/model/inference.py`**

| Name | What it does |
|---|---|
| `normalise(text)` | NFKC, collapse whitespace, drop quote characters, casefold |
| `truncate_document(text, max_chars)` | Head two thirds + marker + tail third; returns the text and a flag |
| `build_prompt(document)` | Rules + JSON schema + delimited document |
| `verify_field(name, value, evidence, document)` | Status and reason, following the table in Method |
| `extract_fields(text)` | One extraction end to end → `ExtractResponse` |

**`src/api/`**

| Name | What it does |
|---|---|
| `main.lifespan()` | Configures logging and calls `ensure_model_ready()` before serving |
| `routes.health()`, `get_schema()`, `extract()`, `list_dataset()`, `demo()` | The five endpoints |
| `schemas.FieldStatus`, `FieldResult`, `ExtractRequest`, `ExtractResponse` | Field verdicts and the extraction request/response |
| `schemas.HealthResponse`, `SchemaResponse`, `DatasetItem`, `DemoResponse` | Bodies of the other endpoints |

**`src/dataset.py`**

| Name | What it does |
|---|---|
| `load_test_set()` | Reads `labels.json` and the `.txt` files once; empty mapping if the data is missing |

**`scripts/evaluate.py`**

| Name | What it does |
|---|---|
| `norm(value)` | Lenient comparison text: NFKC, casefold, punctuation to spaces |
| `matches(field, pred, gold)` | Exact for dates and type, name containment for parties, every jurisdiction for governing law, containment for renewal |
| `classify(field, pred, gold)` | `correct`, `correct_null`, `missed`, `wrong` or `hallucinated` |
| `rate()`, `cell()`, `to_markdown()` | Ratio helper and the markdown report |
| `main()` | Reads the field list from `GET /schema`, POSTs each document, scores every field, writes `results/results.json` and `results/results.md` |

**`tests/test_smoke_ollama.py`**

| Name | What it does |
|---|---|
| `raw_response` | One real structured call on a contract snippet, shared by the tests |
| `test_output_parses_into_schema` | The output validates against `ContractExtraction` |
| `test_present_field_filled_absent_field_null` | Governing law is found; the missing expiration date is null |
| `test_evidence_generated_before_value` | `evidence` precedes `value` in every field |
