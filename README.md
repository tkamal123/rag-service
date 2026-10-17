# Production RAG Service

A retrieval-augmented question-answering API that is **measured, guarded, containerised and deployable**: FastAPI + ChromaDB + any OpenAI-compatible LLM (free Gemini/Groq tiers work; Claude supported too), evaluated with RAGAS, tested in CI with GitHub Actions, and deployed on GCP Cloud Run.

> Ships with a small fictional corpus (`data/docs/`, "Acme Analytics") so it runs out of the box. Replace it with your own `.md`/`.txt` files and update `eval/eval_set.json`.

## Architecture

```
            ┌────────────────────────── FastAPI (app/main.py) ──────────────────────────┐
 question → │ API key → rate limit → input guardrails (length, prompt-injection)          │
            │      → PII redaction → retrieve top-k (Chroma, cosine) → confidence check   │
            │      → LLM (answer only from context, cite [n]) → output PII redaction   │
            └──────────────────────────────────────────────────────────────────────────┘ → answer + sources
 data/docs → ingest/ingest.py (chunk → embed → Chroma, baked into the Docker image)
 eval/     → hit-rate/MRR + refusal rates + RAGAS (faithfulness, relevancy, precision, recall)
```

## Guardrails

| Layer | What it does |
|---|---|
| Input validation | Rejects empty/oversized questions and common prompt-injection phrasing (HTTP 400) |
| PII redaction | Emails, card-like numbers and phone numbers are masked in the question *and* the answer |
| Retrieval confidence | If the best chunk is farther than `MAX_DISTANCE`, the API refuses without calling the LLM (cheaper and avoids guessing) |
| Grounded prompting | Model must answer only from retrieved passages, treats them as untrusted data, and says "I don't know" otherwise |
| Access control | Optional `X-API-Key` header (constant-time compare) and a per-IP rate limit |
| Failure handling | Provider errors return a generic 502; no internals leak |

## Choose an LLM (free options)

Set `LLM_PROVIDER` and the matching key. No credit card is needed for the first three.

| `LLM_PROVIDER` | Key variable | Get a key | Notes |
|---|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` | Google AI Studio → Get API key | Free tier; Google may use free-tier prompts to improve its products, so don't send confidential data |
| `groq` | `GROQ_API_KEY` | console.groq.com | Free tier, very fast; may ask for phone verification |
| `openrouter` | `OPENROUTER_API_KEY` | openrouter.ai | Free `:free` models, small daily cap |
| `ollama` | none | ollama.com | Runs locally; no key or limits, needs RAM |
| `anthropic` | `ANTHROPIC_API_KEY` | console.anthropic.com | Paid |

Free-tier limits and model names change. If a model name errors, pick a current one from the provider's docs and set `LLM_MODEL`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows cmd: .venv\Scripts\activate.bat
pip install -r requirements-dev.txt
python -m pytest -q                                   # 39 tests, no API key needed

export GEMINI_API_KEY=your-key                        # Windows cmd: set GEMINI_API_KEY=your-key
python -m ingest.ingest                               # build the vector index
uvicorn app.main:app --reload                         # http://localhost:8000/docs

curl -s localhost:8000/health                         # shows provider, model and llm_configured
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "How long do monthly customers have to request a refund?"}'
```

## Evaluation

```bash
pip install -r requirements-eval.txt

python -m ingest.ingest                          # baseline index (chunk 500, overlap 80)
python -m eval.run_eval --label baseline         # retrieval + guardrails + RAGAS (uses your LLM key)
python -m eval.run_eval --label baseline --retrieval-only   # free, no API key

python -m ingest.ingest --chunk-size 300 --overlap 50       # change ONE thing
python -m eval.run_eval --label chunk300
python -m eval.compare                           # markdown table of all runs
```

Metrics: **hit-rate@k / MRR** (is the right document retrieved, and how high), **correct/false refusal rate** (does it decline unanswerable questions without declining answerable ones), and RAGAS **faithfulness**, **answer relevancy**, **context precision**, **context recall** (judged by an LLM against reference answers).

On a free tier the RAGAS run is slow (it makes a few hundred small calls) and may hit per-minute limits; the script runs 2 workers with retries (`EVAL_WORKERS` changes this). A small judge model can fail to parse some samples; the result JSON reports how many samples scored (`*_valid_samples`). The retrieval-only run is instant and free.

### Results

<!-- Paste the output of `python -m eval.compare` here. -->

| label | chunk_size | top_k | hit_rate | mrr | faithfulness | answer_relevancy | context_precision | context_recall | false_refusal | correct_refusal |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 500 | 4 | | | | | | | | |
| (your change) | | | | | | | | | | |

## Deploy to GCP Cloud Run

```bash
export PROJECT=your-project-id REGION=europe-west1     # pick the region nearest you
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# secrets live in Secret Manager, never in the image
printf %s "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
printf %s "$(openssl rand -hex 24)" | gcloud secrets create rag-api-key --data-file=-
SA="$(gcloud projects describe $PROJECT --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for s in gemini-api-key rag-api-key; do
  gcloud secrets add-iam-policy-binding $s --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done

gcloud run deploy rag-service --source . --region $REGION --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 2 --timeout 60 \
  --set-env-vars LLM_PROVIDER=gemini \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,API_KEY=rag-api-key:latest

URL=$(gcloud run services describe rag-service --region $REGION --format='value(status.url)')
curl -s $URL/health
curl -s $URL/query -H "X-API-Key: $(gcloud secrets versions access latest --secret rag-api-key)" \
  -H 'Content-Type: application/json' -d '{"question": "What encryption is used for customer data?"}'
```

Cost controls: scale to zero, max 2 instances, API-key gate and rate limit on `/query`. Also set a budget alert in the GCP console.

## CI (GitHub Actions)

- `ci.yml`: on every push/PR runs the test suite, builds the Docker image (which also builds the vector index) and smoke-tests `/health` in the container.
- `eval.yml`: manual run of the RAGAS evaluation (uses your LLM key). Add `GEMINI_API_KEY` (or the key for your provider) under Settings → Secrets → Actions, then run it from the Actions tab and download the result JSON.

## Design decisions and limits

- **Index baked into the image**: fast cold starts and no external database to run. Trade-off: re-ingesting means rebuilding the image. For a large or frequently changing corpus, move to a managed vector DB.
- **Rate limiting is per instance, in memory.** With several Cloud Run instances you would need Redis or an API gateway.
- **Injection/PII guardrails are pattern-based**: a sensible first layer, not a complete defence. Add a classifier or a dedicated guardrails library for higher-risk use.
- **`MAX_DISTANCE` is corpus- and model-specific**: tune it using the eval set's answerable/unanswerable questions.
- **Tests use a fake embedder and fake LLM** so CI is free and deterministic; the real model is exercised by the eval workflow.
