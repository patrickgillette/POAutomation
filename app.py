from math import log
import os, sys, glob, base64, requests, subprocess,json, re, time, shutil, threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr
import jde_orch_client as joc
from jde_orch_client import JDEAuthManager, is_jde_available
from contextlib import contextmanager
from ProcessIndex import ProcessIndexSqlServer, PDFInfo
from config import InitializeEnv
from watch_config import load_watch_state

reader = easyocr.Reader(['en'])

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(lines, (list, tuple)):
        to_write = [
            line if line.endswith("\n") else line + "\n"
            for line in lines
        ]
    else:
        line = str(lines)
        to_write = [line if line.endswith("\n") else line + "\n"]
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(to_write)

#preprocess image to remove highlights and clean up for OCR
def preprocess_image(path, out_path = None,
                                      sat_thresh = 60,    # 0..255 (HSV S)
                                      val_thresh = 60,    # 0..255 (HSV V)
                                      gray_thresh = 180,  # 0..255 luminance to keep ink
                                      do_morph = True):
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

def which(cmd):
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

def find_pdftoppm():
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

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_directories():
    ensure_dir(WATCH_DIR_BASE)
    ensure_dir(PROCESSED_DIR)
    ensure_dir(ERROR_DIR)

def pdf_to_pngs(pdf_path, out_dir, prefix, dpi = 400):
    ensure_dir(out_dir)
    pdftoppm = find_pdftoppm()
    cmd = [pdftoppm, "-png", "-r", str(dpi), pdf_path, prefix]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=out_dir)


    def page_num(p):
        stem = Path(p).stem
        try:
            return int(stem.split("-")[-1])
        except Exception as e:
            return 10**9
    files = sorted(
        glob.glob(str(Path(out_dir) / f"{prefix}-*.png"), recursive=False),
        key=page_num
    )
    if not files:
        raise RuntimeError("No PNGs were generated. Check the PDF/path/dpi.")
    return files

def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def run_ocr(path):
    """Extract raw text from an image using EasyOCR."""
    lines = reader.readtext(path, detail=0)
    return "\n".join(lines)

def send_image_and_text_to_llm(png_path, ocr_text, system_prompt):
    prompt = f"""{system_prompt.strip()}

Here is the raw OCR transcription from this page:

{ocr_text}
"""

    # You can expose this via env if you want to tune it
    max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

    url = f"{BASE_URL.rstrip('/')}/generate"

    try:
        # Send multipart/form-data with the image as a file
        with open(png_path, "rb") as f:
            files = {
                "files": (os.path.basename(png_path), f, "image/png"),
            }
            data = {
                "prompt": prompt,
                "max_new_tokens": str(max_new_tokens),
            }
           
            if not is_llm_ready(BASE_URL, timeout=2.0):
                raise RuntimeError("LLM server is not ready (/ready != ready).")

            r = requests.post(url, data=data, files=files, timeout=TIMEOUT)
            r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"LLM HTTP error: {e}") from e

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"LLM returned non-JSON response: {r.text[:500]}") from e

    output_text = data.get("output")
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError(f"LLM returned empty or invalid 'output': {data}")

    return output_text

def parse_llm_json(txt):
    t = txt.strip()
    # strip possible code fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).rstrip("`").strip()
    # best-effort: grab the first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)

def wait_until_file_stable(path,
                           max_seconds = 60.0):
    """
    True when size unchanged for `checks` consecutive intervals and > 0.
    Returns False if path disappears, remains 0 bytes too long, or we exceed max_seconds.
    """
    
    checks = STABLE_CHECKS
    interval = STABLE_CHECK_INTERVAL_SEC
    start = time.time()
    if not Path(path).exists():
        return False
    last_size = None
    stable_count = 0

    while stable_count < checks:
        if time.time() - start > max_seconds:
            return False

        if not Path(path).exists():
            return False

        try:
            size = Path(path).stat().st_size
        except FileNotFoundError:
            return False

        # If file is 0 bytes, bail immediately
        if size == 0:
            return False

        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size


        if stable_count < checks:
            time.sleep(interval)

    return True

def is_llm_ready(base_url, timeout = 2.0):
    try:
        r = requests.get(f"{base_url.rstrip('/')}/ready", timeout=timeout)
        r.raise_for_status()
        j = r.json()
        return bool(j.get("ready")) is True
    except Exception:
        return False

def wait_for_llm_ready(base_url, max_wait_sec = 30.0, poll_every_sec = 1.0, timeout_per_request = 2.0):
    start = time.time()
    while time.time() - start <= max_wait_sec:
        if is_llm_ready(base_url, timeout=timeout_per_request):
            return True
        time.sleep(poll_every_sec)
    return False

def process_single_pdf(pdf_path,custSoldTo,custShipTo, auth_mgr=None):
    if auth_mgr is None:
        # fallback for direct calls, but in watch_loop we always pass it
        print("[INFO] Acquiring JDE token for single PDF processing.")
        session = joc.get_session()
        auth_mgr = JDEAuthManager(session, joc.login_studio_exact_match)
        auth_mgr.refresh()

    jde_session = auth_mgr.session

    """Process one PDF end-to-end. Returns True/False for success."""
    print(f"[INFO] Processing: {pdf_path}")
    ensure_directories()

    # Fresh frames dir per file to avoid mixing pages between runs
    file_stem = Path(pdf_path).stem
    run_frames_dir = str(Path(WORK_DIR) / f"frames_{file_stem}")
    ensure_dir(run_frames_dir)

    # Per-run debug log
    run_log = str(Path(run_frames_dir) / "debug_log.txt")
    write_log(run_log, f"[{timestamp()}] === Start run for {pdf_path} ===")

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
    all_pages_ok = True
    try:
        pngs = pdf_to_pngs(pdf_path, run_frames_dir, OUTPUT_PREFIX, DPI)
        print(f"Generated {len(pngs)} page image(s).")
        write_log(run_log, f"[{timestamp()}] INFO: Generated {len(pngs)} page image(s)")
    except Exception as e:
        print("Conversion failed:", e)
        write_log(run_log, f"[{timestamp()}] ERROR: {e}")
        raise

    for i, png in enumerate(pngs, start=1):
        page_prefix = f"{file_stem}_page{i}"
        page_ok = True
        payload = None
        try:
            preprocessed = preprocess_image(png)
            raw_text = run_ocr(preprocessed)

            #log ocr text
            ocr_path = str(Path(run_frames_dir) / f"{page_prefix}_ocr.txt")
            write_log(ocr_path, raw_text)
            write_log(run_log, f"[{timestamp()}] INFO: OCR written -> {ocr_path}")

            print(f"Sending {Path(png).name} to {BASE_URL} …")

            system_prompt = f"""You are responsible for ingesting POs into an ERP automation pipeline.
                This is a base template with some fields filled in:
                payload = {{
                '    "Sold_To": "{custSoldTo}",'
                '    "Ship_To": "{custShipTo}",'
                '    "CustomerPO": "string",'
                '    "RequiredDate": "string",'
                '    "Items": ['
                '        {{"Quantity_Ordered": "string that is > 1000", "Item_Number": "#string"}}'
                '    ],'
                '    "P5542101_Version": "DS0001"'
                }}
                Please fill in the rest and output only the JSON.
                You may notice Item_Number has a pound sign (#) in front of it. 
                That needs to remain there while you fill in the customers number after.
                The Quantity_Ordered must be the real amount ordered. So if it says 12.5 but the unit is thousands you need to output 12500. Anything under 1000 is certainly an error.
                """

            system_prompt_log_path = str(Path(run_frames_dir) / f"{file_stem}_system_prompt_page{i}.txt")
            write_log(system_prompt_log_path, system_prompt)
            write_log(run_log, f"[{timestamp()}] INFO: System prompt written -> {system_prompt_log_path}")

            raw = send_image_and_text_to_llm(preprocessed, raw_text,system_prompt)

            llm_raw_path = str(Path(run_frames_dir) / f"{file_stem}_llm_raw_page{i}.txt")
            write_log(llm_raw_path, raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2))
            write_log(run_log, f"[{timestamp()}] INFO: LLM raw output written -> {llm_raw_path}")

            content = (
                raw.removeprefix("```json\n")
                   .removesuffix("```")
                   .strip()
            )
            print("LLM cleaned response:", content)

            write_log(run_log, f"[{timestamp()}] DEBUG: LLM cleaned content length = {len(content)}")

            payload = parse_llm_json(content)

            llm_parsed_path = str(Path(run_frames_dir) / f"{file_stem}_llm_parsed_page{i}.json")
            write_log(llm_parsed_path, payload)
            write_log(run_log, f"[{timestamp()}] INFO: LLM parsed JSON written -> {llm_parsed_path}")

        except Exception as e:
            # Any LLM / parsing problem comes here
            page_ok = False
            all_pages_ok = False  # mark entire PDF as not fully successful
            err_msg = f"[LLM/PARSE ERROR] {Path(png).name}: {e}"
            print(err_msg)
            write_log(run_log, f"[{timestamp()}] ERROR: {err_msg}")

            # Build a fallback payload that will cause JDE to error,
            # but still give it useful context for troubleshooting.
            payload = {
                "Sold_To": custSoldTo,
                "Ship_To": custShipTo,
                "CustomerPO": f"LLM_ERROR_{file_stem}_PAGE_{i}",
                "RequiredDate": "",
                "Items": [],
                "P5542101_Version": "DS0001",
                "LLM_Error": str(e),
                "Pdf_File": str(pdf_path),
                "Pdf_Page": i,
            }

        try:
            r = joc.call_orch(
                local_session,
                local_token,
                orch_name=joc.ORCH_NAME,
                payload=payload,
                wrap_inputs=joc.WRAP_INPUTS,
                timeout=60
            )
            if r is None:
                page_ok = False
                all_pages_ok = False
                print(f"[ORCH ERROR] {Path(png).name}: orchestrator returned None")
                write_log(run_log, f"[{timestamp()}] ERROR: ORCH returned None")
            else:
                print(f"[ORCH] {Path(png).name} -> HTTP {r.status_code}")
                try:
                    print(json.dumps(r.json(), indent=2))
                except Exception as e:
                    print(r.text)

                if not (200 <= r.status_code < 300):
                    page_ok = False
                    all_pages_ok = False
                    write_log(run_log, f"[{timestamp()}] ERROR: ORCH HTTP {r.status_code}")

        except Exception as e:
            page_ok = False
            all_pages_ok = False
            msg = f"[ORCH CALL ERROR] {Path(png).name}: {e}"
            print(msg)
            write_log(run_log, f"[{timestamp()}] ERROR: {msg}")

        if page_ok:
            print(f"[OK] {Path(png).name}")
        else:
            print(f"[PAGE FAILED] {Path(png).name}")

    out_txt = Path(run_frames_dir) / "output.txt"
    write_log(out_txt, "\n\n".join(combined_sections) if combined_sections else "_No output._\n")

    # Only return True if *every* page had successful LLM+ORCH
    return all_pages_ok

def copy_safely(src, dst_dir):
    ensure_dir(dst_dir)
    base = Path(src).name
    target = Path(dst_dir) / base

    # Avoid overwrite collisions
    if target.exists():
        target = target.with_stem(f"{target.stem}_{int(time.time())}")

    try:
        shutil.copy2(src, target)
    except :
        write_log(str(Path(dst_dir) / "error_log.txt"), f"[{timestamp()}] ERROR: Could not copy {src} to {target}")
    
    return str(target)


def log_watch_dirs(watch_dirs):
    for d in watch_dirs:
        print(f"[TEST] Watching: {d['watch_dir']} (Customer: {d.get('customer')}, Address: {d.get('address')})")


def init_auth_manager():
    """Initialize and warm up the JDE auth manager."""
    jde_session = joc.get_session()
    auth_mgr = JDEAuthManager(
        session=jde_session,
        login_fn=joc.login_studio_exact_match,
        ttl_seconds=50 * 60,
        refresh_margin=5 * 60,
    )
    try:
        auth_mgr.refresh()
        print("[DEBUG] Startup JDE token acquired.")
    except Exception as e:
        print(f"[WARN] Could not obtain AIS token at startup: {e}. Will retry on first job.")
    return auth_mgr

def discover_files(idx, watch_dir, customerNumber, addressNumber):
    disk_pdfs = [p for p in glob.glob(str(Path(watch_dir) / "*.pdf")) if p.lower().endswith(".pdf")]
    disk_pdfs = [p for p in disk_pdfs if Path(p).parent.samefile(watch_dir)]
    print(f"[WATCH] Found {len(disk_pdfs)} PDF(s) on disk at {timestamp()} at {watch_dir}.")
    for full_path in disk_pdfs:
        idx.upsert_discovered(full_path, customerNumber=customerNumber, addressNumber=addressNumber)

#Main loop
def watch_loop():
    ensure_directories()
    idx = ProcessIndexSqlServer(SQL_CONN_STR)
    auth_mgr = init_auth_manager()

    while True:
        try:
            WATCH_DIRS, cust_dir_map = load_watch_state(CONFIG_PATH, WATCH_DIR_BASE)
            log_watch_dirs(WATCH_DIRS)

            if not is_jde_available(auth_mgr):
                print("[WATCH] JDE unavailable; not queueing or processing PDFs this cycle.")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            if not wait_for_llm_ready(BASE_URL, max_wait_sec=10, poll_every_sec=1, timeout_per_request=2):
                print(f"[WATCH] LLM at {BASE_URL} not ready; skipping this cycle.")
                time.sleep(POLL_INTERVAL_SEC)
                continue
            else:
                print("[WATCH] LLM is ready.")

            for d in WATCH_DIRS:
                discover_files(idx, d["watch_dir"], d.get("customer"), d.get("address"))

            candidates = idx.get_unprocessed_candidates()
            print(f"[WATCH] {len(candidates)} unprocessed candidate(s).")

            for pdf in candidates:
                if not is_jde_available(auth_mgr):
                    print("[WATCH] JDE became unavailable mid-run; stopping processing loop.")
                    break

                cust = (pdf.customerNumber or "").strip()
                if cust not in cust_dir_map:
                    print(f"[SKIP] No watch_dir configured for customer {cust}; leaving queued: {getattr(pdf, 'file_name', getattr(pdf, 'path', 'UNKNOWN'))}")
                    continue

                file_name = getattr(pdf, "file_name", None) or Path(getattr(pdf, "path")).name
                full_path = str(Path(cust_dir_map[cust]) / file_name)

                # Skip if already success (new signature: (file_name, customer))
                if idx.is_already_processed(full_path, cust):
                    print(f"[SKIP] Already processed successfully: {full_path}")
                    continue

                # Fail missing/0-byte files immediately
                try:
                    if not Path(full_path).exists():
                        idx.mark_processing(full_path, cust)
                        reason = "File not found at expected path; marking as error."
                        print(f"[FAIL] {full_path}: {reason}")
                        idx.mark_result(full_path, customerNumber=cust, success=False, error=reason)
                        continue

                    if Path(full_path).stat().st_size == 0:
                        idx.mark_processing(full_path, cust)
                        reason = "File is 0 bytes; marking as error."
                        print(f"[FAIL] {full_path}: {reason}")
                        idx.mark_result(full_path, customerNumber=cust, success=False, error=reason)
                        continue

                except FileNotFoundError:
                    idx.mark_processing(full_path, cust)
                    reason = "File disappeared before processing; marking as error."
                    print(f"[FAIL] {full_path}: {reason}")
                    idx.mark_result(full_path, customerNumber=cust, success=False, error=reason)
                    continue

                # Ensure the file stops changing; on timeout, fail (don’t loop forever)
                idx.mark_processing(full_path, cust)
                stable = wait_until_file_stable(full_path, max_seconds=MAX_STABILITY_WAIT_SEC)
                if not stable:
                    reason = f"File never became stable within {int(MAX_STABILITY_WAIT_SEC)}s (or became 0 bytes/disappeared)."
                    print(f"[FAIL] {full_path}: {reason}")
                    try:
                        copied_path = copy_safely(full_path, ERROR_DIR)
                    except Exception:
                        copied_path = None
                    idx.mark_result(full_path, customerNumber=cust, success=False, error=reason)
                    continue

                # Ensure we can open (not locked) — otherwise fail
                try:
                    with open(full_path, "rb"):
                        pass
                except Exception as e:
                    reason = f"File locked/unreadable before processing: {e}"
                    print(f"[FAIL] {full_path}: {reason}")
                    try:
                        copied_path = copy_safely(full_path, ERROR_DIR)
                    except Exception:
                        copied_path = None
                    idx.mark_result(full_path, customerNumber=cust, success=False, error=reason)
                    continue

                # Process
                try:
                    ok = process_single_pdf(
                        full_path,
                        custSoldTo=pdf.customerNumber,
                        custShipTo=pdf.addressNumber,
                        auth_mgr=auth_mgr
                    )
                    dst_dir = PROCESSED_DIR if ok else ERROR_DIR
                    copied_path = copy_safely(full_path, dst_dir)
                    idx.mark_result(full_path, customerNumber=cust, success=ok, error=None if ok else "Unknown error")
                except Exception as e:
                    print(f"[FATAL] {full_path}: {e}")
                    try:
                        copied_path = copy_safely(full_path, ERROR_DIR)
                    except Exception:
                        copied_path = None
                    idx.mark_result(full_path, customerNumber=cust, success=False, error=str(e))

            time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\n[WATCH] Stopped by user.")
            break
        except Exception as e:
            print(f"[WATCH ERROR] {e}")
            time.sleep(POLL_INTERVAL_SEC)

    
if __name__ == "__main__":
    InitializeEnv()
    from config import (
        Environment,SQL_CONN_STR, CONFIG_PATH, WATCH_DIR_BASE,
        PROCESSED_DIR, ERROR_DIR, WORK_DIR,
        DPI, OUTPUT_PREFIX,
        BASE_URL, TIMEOUT,
        STABLE_CHECK_INTERVAL_SEC, STABLE_CHECKS,
        POLL_INTERVAL_SEC, MAX_STABILITY_WAIT_SEC,
        
    )    
    watch_loop()

