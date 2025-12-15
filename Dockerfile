FROM python:3.11-slim

# System deps + unixODBC
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    ca-certificates curl gnupg \
    unixodbc unixodbc-dev \
 && rm -rf /var/lib/apt/lists/*

# Microsoft repo + ODBC Driver 17 for SQL Server
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/microsoft-prod.list \
 && apt-get update \
 && ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends msodbcsql17 \
 && rm -rf /var/lib/apt/lists/*


# App dirs
WORKDIR /app
RUN mkdir -p /data /work/pdf_frames


# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# App code
COPY app.py ./
COPY config.py ./
COPY watch_config.py ./
COPY jde_orch_client.py ./
COPY ProcessIndex.py ./

# Non-root user (safer)
RUN useradd -m runner && chown -R runner:runner /app /data /work
USER runner

# Default command
CMD ["python", "-u", "app.py"]