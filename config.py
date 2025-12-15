import os

Environment = "Production"  

# Module-level variables
SQL_CONN_STR = None
CONFIG_PATH = None
WATCH_DIR_BASE = None
PROCESSED_DIR = None
ERROR_DIR = None
WORK_DIR = None

DPI = None
OUTPUT_PREFIX = None

BASE_URL = None
API_KEY = None
TIMEOUT = None

STABLE_CHECK_INTERVAL_SEC = None
STABLE_CHECKS = None
POLL_INTERVAL_SEC = None
MAX_STABILITY_WAIT_SEC = None


_INITIALIZED = False


def InitializeEnv():
    global Environment
    global _INITIALIZED
    global SQL_CONN_STR
    global CONFIG_PATH, WATCH_DIR_BASE, PROCESSED_DIR, ERROR_DIR, WORK_DIR
    global DPI, OUTPUT_PREFIX
    global BASE_URL, TIMEOUT
    global STABLE_CHECK_INTERVAL_SEC, STABLE_CHECKS, POLL_INTERVAL_SEC, MAX_STABILITY_WAIT_SEC


    if _INITIALIZED:
        return  # idempotent;

    # Required
    SQL_CONN_STR = os.getenv("SQL_CONN_STR")
    if not SQL_CONN_STR:
        raise RuntimeError(
            "SQL_CONN_STR env var is required (ODBC connection string for SQL Server)."
        )

    # Paths
    CONFIG_PATH = os.getenv("CONFIG_PATH", "/local_files/config.json")
    WATCH_DIR_BASE = os.getenv("WATCH_DIR_BASE", "/input")
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", "/app/data/processed")
    ERROR_DIR = os.getenv("ERROR_DIR", "/app/data/error")
    WORK_DIR = os.getenv("WORK_DIR", "/app/work/pdf_frames")

    # Rendering / LLM
    DPI = int(os.getenv("DPI", "700"))
    OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "pages")

    BASE_URL = "http://host.docker.internal:8000"
    TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))

    # Stability / polling
    STABLE_CHECK_INTERVAL_SEC = float(os.getenv("STABLE_CHECK_INTERVAL_SEC", "1.0"))
    STABLE_CHECKS = int(os.getenv("STABLE_CHECKS", "3"))
    POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "3.0"))
    MAX_STABILITY_WAIT_SEC = float(os.getenv("MAX_STABILITY_WAIT_SEC", "40"))

    _INITIALIZED = True
