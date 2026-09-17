# Build the model in one stage, ship only what serving needs in the next.
FROM python:3.11-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --target /install . \
 && PYTHONPATH=/install python -m mlserve train --version baked --models /install/models

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MLSERVE_MODELS=/app/models
WORKDIR /app
COPY --from=build /install /app
# run as a non-root user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["python", "-m", "uvicorn", "mlserve.app:app", "--host", "0.0.0.0", "--port", "8000"]
