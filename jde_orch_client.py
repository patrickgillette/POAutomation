import os, json, requests, time
from typing import Any, Dict, Optional

# jde_orch_client.py
# Credentials + device headers
JDE_USER = os.getenv("JDE_USER", "")
JDE_PASS = os.getenv("JDE_PASS", "")
ENV = os.getenv("JDE_ENV", "")
ROLE = os.getenv("JDE_ROLE", "*")
DEVICE = os.getenv("JDE_DEVICE", "PO-Auto-Docker")

JDE_HOST = os.getenv("JDE_HOST", "http://r24uxpd02.dscontainer.local:7005/")
STUDIO_LOGIN = f"{JDE_HOST}/jderest/studio/login"
ORCH_BASES = [f"{JDE_HOST}/jderest/v3/orchestrator"]


ORCH_NAME = os.getenv("JDE_ORCH_NAME", "PG03_ORCH_P5542101CreatePO")
WRAP_INPUTS = os.getenv("JDE_WRAP_INPUTS", "false").lower() == "true"


BASE_HDRS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Accept-Encoding": "identity",
    }


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(BASE_HDRS)
    return s

def login_studio_exact_match(session: requests.Session) -> Optional[Dict[str, Any]]:
    r = session.post(
        STUDIO_LOGIN,
        headers={
        "Accept":"application/json",
        "Content-Type":"application/json",
        "jde-AIS-Auth-Device": DEVICE,
        },
        json={
        "username": JDE_USER,
        "password": JDE_PASS,
        "environment": ENV,
        "role": ROLE,
        },
        timeout=30,
        )
    if r.status_code != 200:
        print("[login] Non-200 from studio/login:", r.status_code, r.text)
        return None
    data = r.json()
    return {
        "token": (data.get("userInfo") or {}).get("token"),
        "session_cookie": data.get("aisSessionCookie"),
        "full_response": data,
        }


# -----------------------
# Orchestrator call
# -----------------------


def call_orch(
    session: requests.Session,
    token: str,
    orch_name: str = ORCH_NAME,
    payload: Optional[Dict[str, Any]] = None,
    wrap_inputs: bool = WRAP_INPUTS,
    timeout: int = 60,
    ) -> Optional[requests.Response]:
    body = {} if payload is None else ({"inputs": payload} if wrap_inputs else payload)
    for base in ORCH_BASES:
        url = f"{base}/{orch_name.lstrip('/')}"
        hdrs = {**BASE_HDRS, "JDE-AIS-Auth": token, "JDE-AIS-Auth-Device": DEVICE}
        r = session.post(url, headers=hdrs, json=body, timeout=timeout)
        if r.status_code == 401:
            hdrs_lower = {**BASE_HDRS, "jde-AIS-Auth": token, "jde-AIS-Auth-Device": DEVICE}
            r = session.post(url, headers=hdrs_lower, json=body, timeout=timeout)
        if r.status_code == 404:
            print(f"[orchestrator] {url} - 404; trying next base…")
            continue
        return r
    print("All orchestrator paths returned 404.")
    return None

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

def is_jde_available(auth_mgr: "JDEAuthManager") -> bool:
    """
    Return True if we can obtain a valid JDE token.
    Do NOT throw; just log and return False on failure.
    """
    try:
        _ = auth_mgr.get_token()
        return True
    except Exception as e:
        print(f"[WATCH] JDE token unavailable: {e}")
        return False