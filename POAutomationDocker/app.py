from math import log
import os, sys, glob, base64, requests, subprocess,json, re, time, shutil, threading
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr
import jde_orch_client as joc
import sqlite3
from contextlib import contextmanager

# ===============Config===========================
# Folders can be overridden by env vars; defaults are Linux-/Docker-friendly
WATCH_DIR = os.getenv("WATCH_DIR", "/data")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "/app/data/processed")
ERROR_DIR = os.getenv("ERROR_DIR", "/app/data/error")
WORK_DIR = os.getenv("WORK_DIR", "/app/work/pdf_frames")
INDEX_DB = os.getenv("INDEX_DB", "/app/data/processed_index.sqlite")


# Rendering / LLM settings
DPI = int(os.getenv("DPI", "700"))
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "pages")


BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:3000")
MODEL = os.getenv("LLM_MODEL", "gemma3:27b")
API_KEY = os.getenv("LLM_API_KEY", "")
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))


STABLE_CHECK_INTERVAL_SEC = float(os.getenv("STABLE_CHECK_INTERVAL_SEC", "1.0"))
STABLE_CHECKS = int(os.getenv("STABLE_CHECKS", "3"))
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "3.0"))
# ===================================================

headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

reader = easyocr.Reader(['en'])
"""
def preprocess_image(path: str, out_path: str | None = None):   
    img = Image.open(path)
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    out_path = out_path or path
    img.save(out_path, "PNG")
    return out_path
"""

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_append(path: str, *lines: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")

def write_text(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def write_json(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _fingerprint(path: str) -> dict:
    """Stable fingerprint for idempotency."""
    p = Path(path)
    size = p.stat().st_size
    mtime_ns = p.stat().st_mtime_ns
    import hashlib
    h = hashlib.sha256()
    # Hash the entire file (robust). If PDFs are huge, you can switch to chunked read.
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": size, "mtime_ns": mtime_ns}


class ProcessIndex:
    """
    Central, durable index for file processing.
    Tracks lifecycle: queued -> processing -> success|error
    Ensures idempotency based on (path, sha256, size, mtime_ns).
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','processing','success','error')),
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                audit_copy TEXT,
                discovered_at TEXT NOT NULL,
                processing_started_at TEXT,z
                processed_at TEXT
            );
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_files_sha ON files(sha256)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)")
            con.commit()

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield con
        finally:
            con.close()

    @staticmethod
    def fingerprint(path: str) -> dict:
        p = Path(path)
        size = p.stat().st_size
        mtime_ns = p.stat().st_mtime_ns
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return {"sha256": h.hexdigest(), "size": size, "mtime_ns": mtime_ns}

    def upsert_discovered(self, path: str):
        fp = self.fingerprint(path)
        with self._conn() as con:
            cur = con.cursor()
            # If same exact file+meta exists and is success, treat as already processed.
            existing = cur.execute("SELECT sha256,size,mtime_ns,status FROM files WHERE path=?",
                                   (path,)).fetchone()
            now = ts()
            if existing:
                sha, size, mt, status = existing
                if (sha, size, mt) == (fp["sha256"], fp["size"], fp["mtime_ns"]):
                    # no change; leave status as-is
                    return status
                # File changed at same path; reset record to queued
                cur.execute("""
                    UPDATE files 
                    SET sha256=?, size=?, mtime_ns=?, status='queued', last_error=NULL, audit_copy=NULL, 
                        attempts=0, discovered_at=?, processing_started_at=NULL, processed_at=NULL
                    WHERE path=?
                """, (fp["sha256"], fp["size"], fp["mtime_ns"], now, path))
            else:
                cur.execute("""
                    INSERT INTO files(path, sha256, size, mtime_ns, status, discovered_at) 
                    VALUES(?,?,?,?, 'queued', ?)
                """, (path, fp["sha256"], fp["size"], fp["mtime_ns"], now))
            con.commit()
            return "queued"

    def is_already_processed(self, path: str) -> bool:
        with self._conn() as con:
            row = con.execute("SELECT status FROM files WHERE path=?", (path,)).fetchone()
            return bool(row and row[0] == "success")

    def mark_processing(self, path: str):
        with self._conn() as con:
            con.execute("""
                UPDATE files SET status='processing', processing_started_at=?, attempts=attempts+1
                WHERE path=?
            """, (ts(), path))
            con.commit()

    def mark_result(self, path: str, success: bool, audit_copy: str | None, error: str | None):
        with self._conn() as con:
            status = "success" if success else "error"
            con.execute("""
                UPDATE files 
                SET status=?, last_error=?, audit_copy=?, processed_at=?
                WHERE path=?
            """, (status, (error or None), (audit_copy or None), ts(), path))
            con.commit()

    def get_unprocessed_candidates(self, watch_dir: str) -> list[str]:
        with self._conn() as con:
            rows = con.execute("""
                SELECT path, status FROM files WHERE status IN ('queued','error','processing')
            """).fetchall()

        wd = Path(watch_dir).resolve()
        existing = []
        for path, status in rows:
            p = Path(path)
            try:
                if p.exists():
                    if p.resolve().parent == wd:
                        existing.append(path)
            except Exception:
                pass
        return existing


#preprocess image to remove highlights and clean up for OCR
def preprocess_image(path: str, out_path: str | None = None,
                                      sat_thresh: int = 60,    # 0..255 (HSV S)
                                      val_thresh: int = 60,    # 0..255 (HSV V)
                                      gray_thresh: int = 180,  # 0..255 luminance to keep ink
                                      do_morph: bool = True):
    """
    1) Convert to HSV and turn any sufficiently saturated pixel to white (kills highlighters).
    2) Convert to luminance and keep only 'dark' ink via threshold.
    3) Optional light morphology to knock out specks.
    """
    im = Image.open(path).convert("RGB")

    hsv = im.convert("HSV")
    h, s, v = [np.array(ch, dtype=np.uint8) for ch in hsv.split()]
    rgb = np.array(im, dtype=np.uint8)

    # Apply a mask to the highlighted pixels first, was having issues with the highlighters throwing off the OCR. 
    sat_mask = s > sat_thresh
    # Also ensure it's not a dark pixel
    not_dark = v > val_thresh
    color_mask = sat_mask & not_dark

    # Set all of those pixels to white
    rgb[color_mask] = 255


    # Perceptual luminance from modified ITU-R 601 luma
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)

    # Anything lighter than threshold -> white; darker -> black
    binary = np.where(lum < gray_thresh, 0, 255).astype(np.uint8)
    bin_img = Image.fromarray(binary, mode="L")


    # Using PIL's min/max filters as a stand-in for morphological open:
    if do_morph:
        # 3x3 minimum (erode) followed by 3x3 maximum (dilate) on a binary image
        bin_img = bin_img.filter(ImageFilter.MinFilter(3))
        bin_img = bin_img.filter(ImageFilter.MaxFilter(3))

    out_path = out_path or path
    bin_img.save(out_path, "PNG")
    return out_path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def which(cmd: str) -> str | None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";")
    for p in paths:
        candidate = Path(p) / cmd
        if candidate.exists():
            return str(candidate)
        for ext in exts:
            c2 = candidate.with_suffix(ext.lower())
            if c2.exists():
                return str(c2)
    return None

def find_pdftoppm() -> str:
    exe = which("pdftoppm")
    if exe:
        return exe
    candidates = [
        r"C:\ProgramData\chocolatey\lib\poppler\tools\pdftoppm.exe",
        r"C:\Program Files\poppler\bin\pdftoppm.exe",
        r"C:\Users\{}\AppData\Local\Programs\poppler\Library\bin\pdftoppm.exe".format(os.getlogin()),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "pdftoppm.exe not found. Install Poppler (e.g., `choco install poppler`) "
        "and ensure pdftoppm.exe is on PATH or update the candidate paths in this script."
    )

def pdf_to_pngs(pdf_path: str, out_dir: str, prefix: str, dpi: int = 400):
    ensure_dir(out_dir)
    pdftoppm = find_pdftoppm()
    cmd = [pdftoppm, "-png", "-r", str(dpi), pdf_path, prefix]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=out_dir)


    def page_num(p):
        stem = Path(p).stem
        try:
            return int(stem.split("-")[-1])
        except Exception:
            return 10**9
    files = sorted(
        glob.glob(str(Path(out_dir) / f"{prefix}-*.png"), recursive=False),
        key=page_num
    )
    if not files:
        raise RuntimeError("No PNGs were generated. Check the PDF/path/dpi.")
    return files

def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def run_ocr(path: str) -> str:
    """Extract raw text from an image using EasyOCR."""
    lines = reader.readtext(path, detail=0)
    return "\n".join(lines)

def send_image_and_text_to_llm(png_path: str, ocr_text: str) -> str:
    SYSTEM_PROMPT_STRING = """You are responsible for ingesting POs into an ERP automation pipeline.
    This is a base template with some fields filled in:\n
                payload = {\n
                '    "Sold_To": "103864",\n'
                '    "Ship_To": "109115",\n'
                '    "CustomerPO": "string",\n'
                '    "RequiredDate": "string",\n'
                '    "Items": [\n'
                '        {"Quantity_Ordered": "string that is > 1000", "Item_Number": "#string"}\n'
                '    ],\n'
                '    "P5542101_Version": "DS0001"\n'
                }\n\n
                Please fill in the rest and output only the JSON.
                You may notice Item_Number has a pound sign (#) in front of it. 
                That needs to remain there while you fill in the customers number after.
                The Quantity_Ordered must be the real amount ordered. So if it says 12.5 but the unit is thousands you need to output 12500. Anything under 1000 is certainly an error.
                """
    USER_PROMPT_STRING = f"Here is the raw OCR transcription:\n{ocr_text}\n"
    payload = {
    "model": MODEL,
    "format": "json",  #gemma can handle this, im not sure about others.
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT_STRING},
        # Maybe need to add few shot examples if the model isnt getting it 100%, can use role:developer or maybe role:example?
        {"role": "user", "content": USER_PROMPT_STRING, "images": [b64_image(png_path)]}
    ],
    "stream": False,
    # maybe also use temperature, top_p
}
    url = f"{BASE_URL}/ollama/api/chat"
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")



def parse_llm_json(txt: str) -> dict:
    """
    Accepts raw LLM content (may include ```json fences).
    Returns a Python dict or raises ValueError.
    """
    t = txt.strip()
    # strip possible code fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).rstrip("`").strip()
    # best-effort: grab the first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)


def wait_until_file_stable(path: str,
                           checks: int = STABLE_CHECKS,
                           interval: float = STABLE_CHECK_INTERVAL_SEC,
                           max_seconds: float = 60.0) -> bool:
    """
    True when size unchanged for `checks` consecutive intervals and > 0.
    Returns False if path disappears, remains 0 bytes too long, or we exceed max_seconds.
    """
    start = time.time()
    if not Path(path).exists():
        return False
    last_size = -1
    stable_count = 0
    zero_byte_count = 0

    while stable_count < checks:
        if not Path(path).exists():
            return False

        try:
            size = Path(path).stat().st_size
        except FileNotFoundError:
            return False

        # If file is 0 bytes, count and bail after a few tries
        if size == 0:
            zero_byte_count += 1
            if zero_byte_count >= checks:
                return False
        else:
            zero_byte_count = 0

        if size == last_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size

        if time.time() - start > max_seconds:
            return False

        time.sleep(interval)
    return True
def process_single_pdf(pdf_path: str, auth_mgr=None):
    if auth_mgr is None:
        # fallback for direct calls, but in watch_loop we always pass it
        session = joc.get_session()
        auth_mgr = JDEAuthManager(session, joc.login_studio_exact_match)
        auth_mgr.refresh()

    jde_session = auth_mgr.session

    """Process one PDF end-to-end. Returns True/False for success."""
    print(f"[INFO] Processing: {pdf_path}")
    ensure_dir(PROCESSED_DIR)
    ensure_dir(ERROR_DIR)
    ensure_dir(WORK_DIR)

    # Fresh frames dir per file to avoid mixing pages between runs
    file_stem = Path(pdf_path).stem
    run_frames_dir = str(Path(WORK_DIR) / f"frames_{file_stem}")
    ensure_dir(run_frames_dir)

    # Per-run debug log
    run_log = str(Path(run_frames_dir) / "debug_log.txt")
    log_append(run_log, f"[{ts()}] === Start run for {pdf_path} ===")

    # Login once per run if session/token not provided
    local_session = jde_session or joc.get_session()
    local_token = auth_mgr.get_token() if auth_mgr else None
    if not local_token:
        auth = joc.login_studio_exact_match(local_session)
        if not auth or not auth.get("token"):
            print("Could not obtain AIS token from studio/login endpoint")
            raise RuntimeError("JDE AIS token unavailable")
        local_token = auth["token"]
        if auth.get("session_cookie"):
            try:
                local_session.cookies.set("JSESSIONID", auth["session_cookie"].split("!")[0])
            except Exception:
                pass
        print("[DEBUG] JDE token acquired.")

    combined_sections = []
    try:
        pngs = pdf_to_pngs(pdf_path, run_frames_dir, OUTPUT_PREFIX, DPI)
        print(f"Generated {len(pngs)} page image(s).")
        log_append(run_log, f"[{ts()}] INFO: Generated {len(pngs)} page image(s)")
    except Exception as e:
        print("Conversion failed:", e)
        log_append(run_log, f"[{ts()}] ERROR: {e}")
        raise

    for i, png in enumerate(pngs, start=1):
        page_prefix = f"{file_stem}_page{i}"
        try:
            preprocessed = preprocess_image(png)
            raw_text = run_ocr(preprocessed)

            #log ocr text
            ocr_path = str(Path(run_frames_dir) / f"{page_prefix}_ocr.txt")
            write_text(ocr_path, raw_text)
            log_append(run_log, f"[{ts()}] INFO: OCR written -> {ocr_path}")

            print(f"Sending {Path(png).name} to {BASE_URL} …")
            raw = send_image_and_text_to_llm(preprocessed, raw_text)

            #log llm raw output
            llm_raw_path = str(Path(run_frames_dir) / f"{file_stem}_llm_raw_page{i}.txt")
            write_text(llm_raw_path, raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2))
            log_append(run_log, f"[{ts()}] INFO: LLM raw output written -> {llm_raw_path}")

            content = (
                raw.removeprefix("```json\n")
                   .removesuffix("```")
                   .strip()
            )
            print("LLM cleaned response:", content)
            #log cleaned llm output
            log_append(run_log, f"[{ts()}] DEBUG: LLM cleaned content length = {len(content)}")

            try:
                payload = parse_llm_json(content)

                                # Save parsed JSON payload
                llm_parsed_path = str(Path(run_frames_dir) / f"{file_stem}_llm_parsed_page{i}.json")
                write_json(llm_parsed_path, payload)
                log_append(run_log, f"[{ts()}] INFO: LLM parsed JSON written -> {llm_parsed_path}")


            except Exception as pe:
                msg = f"[PARSE ERROR] {Path(png).name}: {pe}"
                print(msg)
                log_append(run_log, f"[{ts()}] ERROR: {msg}")
                combined_sections.append(f"\n{content}\n")
                continue

            r = joc.call_orch(
                local_session,
                local_token,
                orch_name=joc.ORCH_NAME,
                payload=payload,
                wrap_inputs=joc.WRAP_INPUTS,
                timeout=60
            )
            if r is None:
                print(f"[ORCH ERROR] {Path(png).name}: orchestrator returned None")
            else:
                print(f"[ORCH] {Path(png).name} -> HTTP {r.status_code}")
                try:
                    print(json.dumps(r.json(), indent=2))
                except Exception:
                    print(r.text)

            print(f"[OK] {Path(png).name}")
            combined_sections.append(f"\n{json.dumps(payload, ensure_ascii=False)}\n")

        except requests.HTTPError as he:
            print(f"[HTTP ERROR] {Path(png).name}: {he.response.status_code} {he.response.text}")
        except Exception as e:
            print(f"[ERROR] {Path(png).name}: {e}")

    out_txt = Path(run_frames_dir) / "output.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_sections) if combined_sections else "_No output._\n")
    print(f"[DONE] Combined output written to: {out_txt}")

    # If we made it here, treating as success. Orchestration will pass notifications if a page fails. 
    return True

def copy_safely(src: str, dst_dir: str) -> str:
    ensure_dir(dst_dir)
    base = Path(src).name
    target = Path(dst_dir) / base

    # Avoid overwrite collisions
    if target.exists():
        target = target.with_stem(f"{target.stem}_{int(time.time())}")

    try:
        shutil.copy2(src, target)
    except :
        log_append(str(Path(dst_dir) / "error_log.txt"), f"[{ts()}] ERROR: Could not copy {src} to {target}")
    
        

    return str(target)

#Main loop
def watch_loop():
    ensure_dir(WATCH_DIR)
    ensure_dir(PROCESSED_DIR)
    ensure_dir(ERROR_DIR)

    print(f"[WATCH] Monitoring: {WATCH_DIR}")
    idx = ProcessIndex(INDEX_DB)

    jde_session = joc.get_session()
    auth_mgr = JDEAuthManager(
        session=jde_session,
        login_fn=joc.login_studio_exact_match,
        ttl_seconds=50*60,
        refresh_margin=5*60
    )
    try:
        auth_mgr.refresh()
        print("[DEBUG] Startup JDE token acquired.")
    except Exception as e:
        print(f"[WARN] Could not obtain AIS token at startup: {e}. Will retry on first job.")

    while True:
        try:
            # Enumerate PDFs in the folder
            disk_pdfs = [p for p in glob.glob(str(Path(WATCH_DIR) / "*.pdf")) if p.lower().endswith(".pdf")]
            disk_pdfs = [p for p in disk_pdfs if Path(p).parent.samefile(WATCH_DIR)]

            # Ensure DB has a 'queued' record (or equivalent) for each discovered file
            for pdf in disk_pdfs:
                idx.upsert_discovered(pdf)

            # Ask DB which items still need attention
            candidates = idx.get_unprocessed_candidates(WATCH_DIR)

            for pdf in candidates:
                # Skip if already success
                if idx.is_already_processed(pdf):
                    continue

                try:
                    if Path(pdf).stat().st_size == 0:
                        print(f"[SKIP] {pdf} is 0 bytes; will retry when it has content.")
                        continue
                except FileNotFoundError:
                        continue

                # Wait until fully written
                if not wait_until_file_stable(pdf):
                    continue

                # Ensure we can open (not locked)
                try:
                    with open(pdf, "rb"):
                        pass
                except Exception:
                    continue

                # Process
                try:
                    idx.mark_processing(pdf)
                    ok = process_single_pdf(pdf, auth_mgr=auth_mgr)
                    dst_dir = PROCESSED_DIR if ok else ERROR_DIR
                    copied_path = copy_safely(pdf, dst_dir)
                    idx.mark_result(pdf, success=ok, audit_copy=copied_path, error=None if ok else "Unknown error")
                except Exception as e:
                    print(f"[FATAL] {pdf}: {e}")
                    try:
                        copied_path = copy_safely(pdf, ERROR_DIR)
                    except Exception:
                        copied_path = None
                    idx.mark_result(pdf, success=False, audit_copy=copied_path, error=str(e))

            time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\n[WATCH] Stopped by user.")
            break
        except Exception as e:
            print(f"[WATCH ERROR] {e}")
            time.sleep(POLL_INTERVAL_SEC)


class JDEAuthManager:
    def __init__(self, session, login_fn, ttl_seconds=50*60, refresh_margin=5*60):
        """
        session: requests.Session used for calls
        login_fn: callable(session) -> {"token": "...", "session_cookie": "..."} like joc.login_studio_exact_match
        ttl_seconds: guessed token lifetime (seconds). If unknown, keep conservative (e.g., 50 min).
        refresh_margin: refresh a bit before TTL to avoid race.
        """
        self.session = session
        self.login_fn = login_fn
        self.ttl_seconds = ttl_seconds
        self.refresh_margin = refresh_margin
        self._token = None
        self._issued_at = 0.0

    def _needs_refresh(self) -> bool:
        if not self._token:
            return True
        now = time.time()
        return (now - self._issued_at) >= max(0, self.ttl_seconds - self.refresh_margin)

    def refresh(self):
        auth = self.login_fn(self.session)
        if not auth or not auth.get("token"):
            raise RuntimeError("JDE AIS token unavailable during refresh")
        self._token = auth["token"]
        self._issued_at = time.time()
        # keep session cookie in sync if provided
        if auth.get("session_cookie"):
            try:
                self.session.cookies.set("JSESSIONID", auth["session_cookie"].split("!")[0])
            except Exception:
                pass

    def get_token(self) -> str:
        if self._needs_refresh():
            self.refresh()
        return self._token

if __name__ == "__main__":
    watch_loop()

