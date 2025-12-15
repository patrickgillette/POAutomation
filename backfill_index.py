import sqlite3
import json
from pathlib import Path
from datetime import datetime
import hashlib

# --- HOST PATHS ---
DB = Path(r"C:\DockerContainers\POAutomation\data\processed_index.sqlite")
ROOT = Path(r"\\dsfile4\shared\CUSTOMERS")
CONFIG_PATH = Path(r"C:\DockerContainers\POAutomationPY\config.json")
# -------------------

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Create table with sha256 as UNIQUE instead of path
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            size INTEGER,
            mtime_ns INTEGER,
            status TEXT NOT NULL CHECK(
                status IN ('queued','processing','success','error','backfilled')
            ),
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            audit_copy TEXT,
            customerNumber TEXT,
            addressNumber TEXT,
            discovered_at TEXT NOT NULL,
            processing_started_at TEXT,
            processed_at TEXT
        );
    """)
    con.commit()

    print(f"[BACKFILL] Scanning: {ROOT}")

    # Load config.json
    if not CONFIG_PATH.exists():
        print(f"[BACKFILL] config.json not found: {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_list = cfg.get("watch_dirs") or []
    configs = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        configs.append({
            "path": path,
            "customer": item.get("customer"),
            "address": item.get("address"),
        })

    # Iterate each configured folder
    for conf in configs:
        subdir = conf["path"].lstrip("/\\")
        base_dir = ROOT / subdir

        if not base_dir.exists():
            print(f"[BACKFILL] Skipping missing directory: {base_dir}")
            continue

        print(f"[BACKFILL] Scanning: {base_dir}")

        for pdf in base_dir.rglob("*.pdf"):
            st = pdf.stat()

            sha = file_sha256(pdf)

            # Use sha256 as the conflict key instead of path
            cur.execute(f"""
                INSERT INTO files (
                    path, sha256, size, mtime_ns, status, attempts,
                    last_error, audit_copy, discovered_at,
                    processing_started_at, processed_at, customerNumber, addressNumber
                )
                VALUES (?, ?, ?, ?, 'backfilled', 0, NULL, NULL, ?, NULL, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET 
                    path=excluded.path,
                    status='backfilled';
            """, (
                str(pdf.resolve()),
                sha,
                st.st_size,
                st.st_mtime_ns,
                timestamp(),
                timestamp(),
                conf['customer'],
                conf['address']
            ))

            print(f"[BACKFILL] {pdf} -> backfilled")

    con.commit()
    con.close()
    print("[BACKFILL] Done.")

if __name__ == "__main__":
    main()