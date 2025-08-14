# tools/update_openapi.py
"""
Generează un OpenAPI "curățat" pentru GPT Actions / integrări externe.

Ce face:
  1) Fetch de la FastAPI local:           <local_base>/openapi.json
  2) Servers -> set doar la HTTPS public:  --server https://*.ngrok-free.app
  3) Asigură operationId pentru fiecare metodă (GET/POST/...)
  4) Normalizează components.schemas să fie dict
  5) Repară requestBody / responses: să aibă schema de tip "object" (când lipsesc sau sunt invalide)
  6) Repară /health și /healthz -> răspuns JSON { "status": "ok" } (schema object)
  7) (opțional) filtrează doar căile /ops/* (safe pentru GPT Actions)
  8) (opțional) injectează un API Key header (securitySchemes + security)

Usage:
  python tools/update_openapi.py \
    --local  http://127.0.0.1:8000 \
    --server https://xxxxx.ngrok-free.app \
    --out    static/openapi.public.json \
    --only_ops \
    --inject_api_key X-API-Key

Exit codes:
  0 -> success
  1 -> fetch/open/write errors
  2 -> argumente invalide
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Dict, Any, Iterable, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger("update_openapi")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


# ----------------------------
# Helpers
# ----------------------------
def _fetch_openapi(local_base: str) -> Dict[str, Any]:
    url = local_base.rstrip("/") + "/openapi.json"
    LOG.info(f"Fetch OpenAPI: {url}")
    req = Request(url, headers={"User-Agent": "openapi-updater"})
    try:
        with urlopen(req, timeout=15) as r:
            data = r.read().decode("utf-8")
        spec = json.loads(data)
        if not isinstance(spec, dict) or "openapi" not in spec:
            raise ValueError("Documentul nu pare OpenAPI valid.")
        return spec
    except URLError as e:
        LOG.error(f"Nu pot descărca {url}: {e}")
        raise
    except Exception as e:
        LOG.error(f"OpenAPI JSON invalid: {e}")
        raise


def _mk_operation_id(method: str, path: str) -> str:
    # ex: path="/ops/refresh_all" + POST -> ops_refresh_all_post
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", path.strip("/"))
    clean = re.sub(r"_+", "_", clean).strip("_") or "root"
    return f"{clean}_{method.lower()}"


def _ensure_operation_ids(spec: Dict[str, Any]) -> None:
    paths = spec.get("paths", {})
    for path, item in (paths or {}).items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method.lower() not in (
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "options",
                "head",
            ):
                continue
            if not isinstance(op, dict):
                continue
            if not op.get("operationId"):
                op["operationId"] = _mk_operation_id(method, path)


def _ensure_components_schemas(spec: Dict[str, Any]) -> None:
    comps = spec.setdefault("components", {})
    schemas = comps.get("schemas")
    if schemas is None or not isinstance(schemas, dict):
        comps["schemas"] = {}  # force dict


def _set_servers(spec: Dict[str, Any], public_server: str) -> None:
    if not public_server.startswith("https://"):
        raise ValueError("Argumentul --server trebuie să fie HTTPS (ex: ngrok).")
    spec["servers"] = [{"url": public_server}]


def _optional_filter_paths(
    spec: Dict[str, Any], keep_prefixes: Optional[Iterable[str]] = None
) -> None:
    if not keep_prefixes:
        return
    src = spec.get("paths") or {}
    new_paths = {}
    for p, v in src.items():
        if any(p.startswith(prefix) for prefix in keep_prefixes):
            new_paths[p] = v
    spec["paths"] = new_paths


def _ensure_json_schema_object(schema: Any) -> Dict[str, Any]:
    """
    GPT Actions cere ca schema pentru requestBody să fie 'type: object'.
    Dacă nu e dict sau e alt tip, o coerționăm la:
      { "type": "object", "properties": {}, "additionalProperties": true }
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    t = schema.get("type")
    if t == "object" or "properties" in schema or "additionalProperties" in schema:
        return schema
    # dacă e "array"/"string"/"number"/"$ref" etc, transformăm într-un obiect generic
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _patch_request_bodies(spec: Dict[str, Any]) -> None:
    """
    Asigură că requestBody/content/*/schema este object.
    """
    paths = spec.get("paths", {}) or {}
    for _path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for _method, op in item.items():
            if not isinstance(op, dict):
                continue
            req = op.get("requestBody")
            if not isinstance(req, dict):
                continue
            content = req.get("content")
            if not isinstance(content, dict):
                # dacă nu există content, punem unul "application/json" gol dar valid
                req["content"] = {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        }
                    }
                }
                continue
            for _ctype, desc in content.items():
                if not isinstance(desc, dict):
                    continue
                sch = desc.get("schema")
                desc["schema"] = _ensure_json_schema_object(sch)


def _patch_responses(spec: Dict[str, Any]) -> None:
    """
    Pentru răspunsuri 2xx cu application/json, asigură schema object.
    """
    paths = spec.get("paths", {}) or {}
    for _path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for _method, op in item.items():
            if not isinstance(op, dict):
                continue
            rsps = op.get("responses")
            if not isinstance(rsps, dict):
                continue
            for code, rdef in rsps.items():
                if not isinstance(rdef, dict):
                    continue
                try:
                    code_int = int(code)
                except Exception:
                    continue
                if 200 <= code_int < 300:
                    content = rdef.get("content")
                    if not isinstance(content, dict):
                        # dacă lipsește, adaugăm un JSON generic object
                        rdef["content"] = {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {},
                                    "additionalProperties": True,
                                }
                            }
                        }
                        continue
                    for _ctype, desc in content.items():
                        if not isinstance(desc, dict):
                            continue
                        sch = desc.get("schema")
                        desc["schema"] = _ensure_json_schema_object(sch)


def _patch_health_endpoints(spec: Dict[str, Any]) -> None:
    """
    Asigură /health și /healthz (dacă există) să aibă 200 JSON object cu {status:string}.
    Evită erorile de validare gen: "object schema missing properties".
    """

    def _ensure_health(path: str) -> None:
        paths = spec.setdefault("paths", {})
        if path not in paths or not isinstance(paths[path], dict):
            return
        # for all methods define 200->application/json->schema object
        for m, op in paths[path].items():
            if m.lower() not in ("get", "post", "head", "options"):
                continue
            if not isinstance(op, dict):
                continue
            rsps = op.setdefault("responses", {})
            r200 = rsps.setdefault("200", {})
            content = r200.setdefault("content", {})
            j = content.setdefault("application/json", {})
            j["schema"] = {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
                "additionalProperties": True,
            }

    _ensure_health("/health")
    _ensure_health("/healthz")


def _inject_api_key_security(spec: Dict[str, Any], header_name: Optional[str]) -> None:
    """
    (opțional) Adaugă:
      components.securitySchemes.api_key -> header <header_name>
      security: [ { api_key: [] } ]
    Dacă header_name e None -> nu injectăm nimic.
    """
    if not header_name:
        return
    comps = spec.setdefault("components", {})
    sec = comps.setdefault("securitySchemes", {})
    sec["api_key"] = {"type": "apiKey", "in": "header", "name": header_name}
    # setăm securitate globală
    spec["security"] = [{"api_key": []}]


def _validate_servers_https(spec: Dict[str, Any]) -> None:
    servers = spec.get("servers") or []
    if not servers:
        raise ValueError("Spec-ul nu are 'servers'.")
    for s in servers:
        u = s.get("url", "")
        if not u.startswith("https://"):
            raise ValueError(f"Server URL nu este HTTPS: {u}")


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="Ex: http://127.0.0.1:8000 (FastAPI local)")
    ap.add_argument(
        "--server", required=True, help="Public HTTPS (ngrok): https://*.ngrok-free.app"
    )
    ap.add_argument(
        "--out",
        default="static/openapi.public.json",
        help="Calea de output (default: static/openapi.public.json)",
    )
    ap.add_argument(
        "--only_ops", action="store_true", help="Expune DOAR rutele care încep cu /ops/"
    )
    ap.add_argument(
        "--inject_api_key",
        default=None,
        help="Nume header API Key (ex: X-API-Key). Dacă lipsește, nu injectează security.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = parse_args(argv)
        spec = _fetch_openapi(args.local)

        # Normalize & patch
        _ensure_components_schemas(spec)
        _ensure_operation_ids(spec)
        _set_servers(spec, args.server)
        _patch_request_bodies(spec)
        _patch_responses(spec)
        _patch_health_endpoints(spec)
        if args.only_ops:
            _optional_filter_paths(spec, keep_prefixes=["/ops/"])
        _inject_api_key_security(spec, args.inject_api_key)

        # sanity check
        _validate_servers_https(spec)

        # Write
        out_path = args.out
        # asigurăm directorul părinte dacă e cazul
        try:
            import os

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        except Exception:
            pass

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        LOG.info(f"Scris: {out_path} (servers -> {args.server})")
        return 0
    except Exception as e:
        LOG.error(f"Eșec generare OpenAPI public: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
