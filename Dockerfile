FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app ingest ./ingest
COPY --chown=app:app data ./data
RUN chown -R app:app /home/app

USER app
# Build the vector index at image build time (also caches the embedding model for this user),
# so containers start fast and Cloud Run can scale to zero without re-indexing.
RUN python -m ingest.ingest

ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
