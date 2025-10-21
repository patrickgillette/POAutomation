FROM python:3.11-slim


# System deps for OpenCV/EasyOCR and Poppler (pdftoppm)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
poppler-utils \
libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
&& rm -rf /var/lib/apt/lists/*


# App dirs
WORKDIR /app
RUN mkdir -p /data /work/pdf_frames


# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# App code
COPY app.py ./
COPY jde_orch_client.py ./


# Non-root user (safer)
RUN useradd -m runner && chown -R runner:runner /app /data /work
USER runner


ENV WATCH_DIR=/data \
WORK_DIR=/work/pdf_frames \
PROCESSED_DIR=/data/processed \
ERROR_DIR=/data/error \
INDEX_DB=/data/processed_index.sqlite \
DPI=700 OUTPUT_PREFIX=pages \
LLM_BASE_URL=http://host.docker.internal:3000 \
LLM_MODEL=gemma3:27b \
LLM_TIMEOUT=180 \
STABLE_CHECK_INTERVAL_SEC=1.0 STABLE_CHECKS=3 POLL_INTERVAL_SEC=3.0


# Default command
CMD ["python", "-u", "app.py"]