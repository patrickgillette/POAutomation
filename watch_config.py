# watch_config.py
import json
from pathlib import Path

def load_watch_state(config_path, watch_dir_base):
    cfg_path = Path(config_path)

    def rel_norm(p):
        return (p or "").strip().lstrip("/\\")  # force relative

    watch_dirs = []
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for item in (cfg.get("watch_dirs") or []):
            if not isinstance(item, dict):
                continue
            rel = rel_norm(item.get("path"))
            cust = (item.get("customer") or "").strip()
            addr = (item.get("address") or "").strip() or None
            if not rel:
                continue
            watch_dirs.append({
                "watch_dir": str(Path(watch_dir_base) / rel),
                "customer": cust or None,
                "address": addr,
                "rel_path": rel,
            })

    if not watch_dirs:
        watch_dirs = [{
            "watch_dir": str(Path(watch_dir_base)),
            "customer": None,
            "address": None,
            "rel_path": "",
        }]

    # Build 1:1 customer -> watch_dir map
    cust_dir_map = {}
    for d in watch_dirs:
        cust = (d.get("customer") or "").strip()
        wdir = d.get("watch_dir")
        if not cust or not wdir:
            continue
        if cust in cust_dir_map and Path(cust_dir_map[cust]) != Path(wdir):
            raise ValueError(f"Multiple watch dirs for customer {cust}: '{cust_dir_map[cust]}' vs '{wdir}'")
        cust_dir_map[cust] = wdir

    return watch_dirs, cust_dir_map
