#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap pentru Assistants API + File Search (Vector Stores):
- creează/reutilizează Vector Store (încărcare & indexare fișiere proiect)
- creează/reutilizează Assistant cu tools: file_search + code_interpreter
- salvează state în scripts/.assistant_state.json
- robust pentru Windows (PowerShell), ASCII-only headers, include/exclude globs

Cerințe:
  pip install openai>=1.30 httpx>=0.27 tiktoken
  export/set OPENAI_API_KEY="sk-...." (ASCII only)

Exemple:
  python scripts/bootstrap_assistant.py
  python scripts/bootstrap_assistant.py --root . --include ".py,.md,.html,.json" --exclude ".venv,.git,models_store,node_modules" --model gpt-4o

Autor: tu + GPT (enterprise)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import fnmatch
import pathlib
from typing import List, Optional, Set
from pathlib import Path

# ---- Dependențe externe
try:
    from openai import OpenAI
except Exception:
    print(
        "[FATAL] Nu pot importa openai. Instalează: pip install openai>=1.30",
        file=sys.stderr,
    )
    raise

from dotenv import load_dotenv

load_dotenv()  # încarcă variabilele din .env în os.environ

STATE_PATH = pathlib.Path("scripts/.assistant_state.json").resolve()

# --- File Search: excluderi + stub-uri pentru CSV ---
EXCLUDE_FILESEARCH_EXTS = {
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
}  # CSV îl tratăm special
CSV_STUB_MAX_ROWS = 30  # câte rânduri din CSV includem în stub
CSV_STUB_DIR = Path(__file__).parent / "_retrieval_stubs"
CSV_STUB_DIR.mkdir(parents=True, exist_ok=True)


# --- Compat layer pentru diferențele de SDK ---
def _get_assistants_api(client):
    """
    În unele versiuni: client.assistants
    În altele (mai vechi / beta): client.beta.assistants
    """
    asst = getattr(client, "assistants", None)
    if asst is not None:
        return asst
    beta = getattr(client, "beta", None)
    if beta is not None:
        asst = getattr(beta, "assistants", None)
        if asst is not None:
            return asst
    raise RuntimeError(
        "SDK-ul OpenAI instalat nu expune nici client.assistants, nici client.beta.assistants. "
        "Fă upgrade cu: pip install -U openai"
    )


def _csv_to_stub_text(csv_path: Path, max_rows: int = CSV_STUB_MAX_ROWS) -> str:
    """
    Produce un rezumat textual pentru File Search dintr-un CSV:
    - meta: nume, mărime
    - shape: (rows, cols)
    - head(n)
    - describe() numeric (dacă există)
    """
    import io
    import csv

    text = io.StringIO()
    print(f"# CSV STUB: {csv_path.name}", file=text)
    try:
        size_kb = csv_path.stat().st_size / 1024.0
        print(f"- Size: {size_kb:.1f} KB", file=text)
    except Exception:
        pass

    try:
        # înceracă cu pandas (dacă e disponibil)
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(csv_path)
            print(f"- Shape: {df.shape}", file=text)
            print(f"- Columns: {list(df.columns)}", file=text)
            print("\n## Head")
            print(df.head(max_rows).to_string(index=False), file=text)
            # describe numeric
            num_cols = df.select_dtypes(include="number")
            if not num_cols.empty:
                print("\n## Describe (numeric)")
                print(num_cols.describe().to_string(), file=text)
        except Exception:
            # fallback minimal fără pandas
            with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
                r = csv.reader(f)
                rows = []
                for i, row in enumerate(r):
                    rows.append(row)
                    if i >= max_rows:
                        break
                if rows:
                    print(f"- Columns (guessed): {rows[0]}", file=text)
                    print("\n## Head")
                    for rr in rows[1:]:
                        print(rr, file=text)
    except Exception as e:
        print(f"[stub-error] {e}", file=text)

    return text.getvalue()


def make_csv_stub(csv_path: Path) -> Path | None:
    """
    Creează fișier .txt cu conținutul de mai sus și returnează calea noului fișier.
    """
    try:
        rel = csv_path.name + ".txt"
        out = CSV_STUB_DIR / rel
        content = _csv_to_stub_text(csv_path)
        out.write_text(content, encoding="utf-8")
        return out
    except Exception:
        return None


# ---------------------------
# Utilities
# ---------------------------
def _ensure_ascii_env(varname: str):
    """
    Dacă variabila are caractere non-ASCII, o ștergem din env ca să evităm
    UnicodeEncodeError în httpx (header-urile OpenAI trebuie să fie ASCII).
    """
    val = os.getenv(varname)
    if not val:
        return
    try:
        val.encode("ascii")
    except UnicodeEncodeError:
        print(
            f"[WARN] {varname} conține non-ASCII ('{val}'). Elimin din environment.",
            file=sys.stderr,
        )
        os.environ.pop(varname, None)


def preflight_env() -> str:
    """
    Verifică OPENAI_API_KEY și sanitizează restul variabilelor relevante.
    Returnează cheia.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("[FATAL] OPENAI_API_KEY nu este setat.", file=sys.stderr)
        sys.exit(1)
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        print(
            "[FATAL] OPENAI_API_KEY conține caractere non-ASCII. Înlocuiește cu o cheie ASCII curată.",
            file=sys.stderr,
        )
        sys.exit(1)

    # sanitize pentru variabile pe care SDK le poate pune în header-uri:
    for v in ("OPENAI_PROJECT", "OPENAI_ORGANIZATION", "OPENAI_ORG_ID"):
        _ensure_ascii_env(v)

    # opțional: suport pentru endpoint self-hosted / proxy
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url and not base_url.lower().startswith(("http://", "https://")):
        print(
            f"[WARN] OPENAI_BASE_URL='{base_url}' pare invalid (trebuie http/https). Ignor.",
            file=sys.stderr,
        )
        os.environ.pop("OPENAI_BASE_URL", None)

    return key


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(d: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


# ---------------------------
# File discovery
# ---------------------------
DEFAULT_INCLUDE = ".py,.md,.txt,.json,.yaml,.yml,.html,.htm,.css,.js,.ts,.tsx,.sql,.ipynb,.csv"
DEFAULT_EXCLUDE = ".git,.venv,__pycache__,node_modules,dist,build,models_store,.idea,.vscode,*.pkl,*.joblib,*.onnx,*.bin,*.pt,*.weights"


def parse_globs(globs_csv: str) -> List[str]:
    if not globs_csv:
        return []
    items = [x.strip() for x in globs_csv.split(",") if x.strip()]
    return items


def should_exclude(path: str, exclude_patterns: List[str]) -> bool:
    # exclude dacă oricare pattern (folder sau glob) potrivește
    parts = pathlib.Path(path).parts
    for patt in exclude_patterns:
        if patt.startswith("*.") or ("*" in patt) or ("?" in patt):
            if fnmatch.fnmatch(path, patt):
                return True
        else:
            # nume folder / fișier exact în path
            if patt in parts:
                return True
    return False


def discover_files(
    root_dir: str | Path,
    include_exts: List[str],
    exclude_patterns: List[str],
    max_files: Optional[int] = None,
) -> List[Path]:
    """
    Descoperă fișiere pentru File Search, cu:
      - include_exts: listă de extensii acceptate (ex: [".py",".md",...])
      - exclude_patterns: foldere/glob-uri de exclus (ex: ".venv,.git,*.pkl")
      - max_files: limită opțională pentru debugging
    Reguli speciale:
      - .csv -> generăm automat un STUB .txt (pentru retrieval) și includem doar stub-ul
      - excludem explicit extensiile din EXCLUDE_FILESEARCH_EXTS
    """
    root = Path(root_dir).resolve()
    out: List[Path] = []

    # normalizare extensii incluse (asigură-te că încep cu punct)
    inc: Set[str] = set()
    for e in include_exts or []:
        e = e.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        inc.add(e.lower())

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        # relativ la root, pentru potriviri exclude/glob
        rel = str(p.relative_to(root))

        # skip dacă se potrivește oricare pattern de excludere
        if should_exclude(rel, exclude_patterns):
            continue

        ext = p.suffix.lower()

        # dacă avem listă de include, respect-o strict
        if inc and ext not in inc:
            # excepție: CSV (dacă nu e în include, nu-l procesăm deloc)
            if ext != ".csv":
                continue

        # CSV -> stub .txt (pentru că File Search nu suportă direct .csv)
        if ext == ".csv":
            stub = make_csv_stub(p)
            if stub and stub.exists():
                out.append(stub)
            # nu includem CSV-ul original
            if max_files and len(out) >= max_files:
                break
            continue

        # alte extensii explicit excluse din retrieval
        if ext in EXCLUDE_FILESEARCH_EXTS:
            continue

        out.append(p)

        if max_files and len(out) >= max_files:
            break

    return out

    """
    Adună fișiere suportate pentru File Search.
    - CSV: generăm stub .txt și încărcăm stub-ul
    - Ignorăm foldere de build/cache/model etc.
    """
    out: list[Path] = []
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.parts)
        if any(bad in parts for bad in {".git", ".venv", "__pycache__", "models_store", "static"}):
            continue

        ext = p.suffix.lower()

        # 1) CSV -> înlocuim cu stub .txt
        if ext == ".csv":
            stub = make_csv_stub(p)
            if stub and stub.exists():
                out.append(stub)
            # nu includem p (CSV-ul brut), doar stub-ul!
            continue

        # 2) Alte extensii explicit excluse din retrieval
        if ext in EXCLUDE_FILESEARCH_EXTS:
            continue

        out.append(p)

    return out


# ---------------------------
# OpenAI helpers
# ---------------------------
def get_client() -> OpenAI:
    key = preflight_env()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    if base_url:
        print(f"[INFO] Folosesc OPENAI_BASE_URL={base_url}")
        client = OpenAI(api_key=key, base_url=base_url)
    else:
        client = OpenAI(api_key=key)
    return client


def create_or_reuse_vector_store(client: OpenAI, name: str, vector_store_id: Optional[str]) -> str:
    if vector_store_id:
        print(f"[OK] Refolosesc Vector Store existent: {vector_store_id}")
        return vector_store_id

    vs = client.vector_stores.create(name=name)
    print(f"[OK] Vector Store creat: {vs.id}")
    return vs.id


def upload_and_index_paths(client: OpenAI, vector_store_id: str, paths: List[pathlib.Path]) -> None:
    if not paths:
        print("[WARN] Nu sunt fișiere de urcat (filtrele includ/exclud au golit lista).")
        return

    # deschide file handles
    files = []
    total_bytes = 0
    for p in paths:
        try:
            f = open(p, "rb")
            files.append((p, f))
            total_bytes += p.stat().st_size
        except Exception as e:
            print(f"[WARN] Sar peste {p}: {e}")

    if not files:
        print("[WARN] Niciun fișier deschis cu succes.")
        return

    print(f"[INFO] Încarc {len(files)} fișiere ({human_bytes(total_bytes)}) în Vector Store...")

    try:
        # upload în batch + poll pentru indexare
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id, files=[fh for _, fh in files]
        )
        print("[OK] Upload & indexare finalizate.")
    finally:
        for _, fh in files:
            try:
                fh.close()
            except Exception:
                pass


def create_or_update_assistant(
    client: OpenAI,
    name: str,
    model: str,
    instructions: str,
    vector_store_id: str,
    assistant_id: Optional[str],
    update_existing: bool,
) -> str:
    asst_api = _get_assistants_api(client)
    tools = [
        {"type": "file_search"},
        {"type": "code_interpreter"},
    ]
    tool_resources = {"file_search": {"vector_store_ids": [vector_store_id]}}

    if assistant_id:
        if update_existing:
            asst = asst_api.update(
                assistant_id,
                name=name,
                model=model,
                instructions=instructions,
                tools=tools,
                tool_resources=tool_resources,
            )
            print(f"[OK] Assistant updatat: {asst.id}")
            return asst.id
        else:
            print(f"[OK] Refolosesc Assistant existent (fără update): {assistant_id}")
            return assistant_id

    asst = asst_api.create(
        name=name,
        model=model,
        instructions=instructions,
        tools=tools,
        tool_resources=tool_resources,
    )
    print(f"[OK] Assistant creat: {asst.id}")
    return asst.id


# ---------------------------
# Main
# ---------------------------
def read_instructions(path: Optional[str]) -> str:
    default = (
        "Ești co-programator pentru un proiect FastAPI + ML trading.\n"
        "- Respectă PEP8, tipare stricte, docstrings.\n"
        "- Când propui patch-uri, furnizează diff unificat și ancorează-l (fișier + linia după care se lipește).\n"
        "- Folosește contextul din Vector Store când răspunzi.\n"
        "- Preferă patch-uri incrementale, fără a rupe structura.\n"
    )
    if not path:
        return default
    p = pathlib.Path(path)
    if not p.exists():
        print(
            f"[WARN] Fișierul de instrucțiuni nu există: {p}. Folosesc default.",
            file=sys.stderr,
        )
        return default
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Nu pot citi {p}: {e}. Folosesc default.", file=sys.stderr)
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bootstrap Assistants API + File Search (Vector Stores)"
    )
    ap.add_argument(
        "--root",
        default=".",
        help="Rădăcina proiectului pentru scanare fișiere (default: .)",
    )
    ap.add_argument(
        "--include",
        default=DEFAULT_INCLUDE,
        help=f"Extensii incluse (CSV). Default: {DEFAULT_INCLUDE}",
    )
    ap.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help=f"Pattern-uri/foldere excluse (CSV). Default: {DEFAULT_EXCLUDE}",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limitează numărul de fișiere (debug).",
    )
    ap.add_argument(
        "--vector-store-id",
        default=None,
        help="Refolosește un Vector Store existent (ID).",
    )
    ap.add_argument("--assistant-id", default=None, help="Refolosește un Assistant existent (ID).")
    ap.add_argument("--name", default="StockApp Engineer", help="Numele Assistant-ului.")
    ap.add_argument(
        "--model",
        default=os.getenv("ASSISTANT_MODEL", "gpt-4o"),
        help="Modelul Assistant-ului (ex: gpt-4o).",
    )
    ap.add_argument("--instructions", default=None, help="Fișier cu instrucțiuni (text).")
    ap.add_argument(
        "--update-existing",
        action="store_true",
        help="Dacă se dă --assistant-id, forțează update-ul acestuia.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Nu urcă fișiere / nu creează resurse, doar arată ce s-ar face.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    state = load_state()

    # --- client
    client = get_client()

    # --- vector store
    vs_id = args.vector_store_id or state.get("vector_store_id")
    if args.dry_run:
        print("[DRY] Aș crea/refolosi Vector Store aici.")
        vs_id = vs_id or "vs_dummy"
    else:
        vs_id = create_or_reuse_vector_store(
            client, name="StockApp Codebase", vector_store_id=vs_id
        )

    # --- files
    include_exts = parse_globs(args.include)
    exclude_patterns = parse_globs(args.exclude)

    files = discover_files(args.root, include_exts, exclude_patterns, args.max_files)
    print(f"[INFO] Găsite {len(files)} fișiere pentru încărcare.")
    if args.dry_run:
        for p in files[:10]:
            print(f"  - {p}")
        if len(files) > 10:
            print(f"  ... și încă {len(files)-10} fișiere.")
    else:
        upload_and_index_paths(client, vs_id, files)

    # --- assistant
    instr = read_instructions(args.instructions)
    asst_id = args.assistant_id or state.get("assistant_id")
    if args.dry_run:
        print("[DRY] Aș crea/refolosi Assistant aici.")
        asst_id = asst_id or "asst_dummy"
    else:
        asst_id = create_or_update_assistant(
            client=client,
            name=args.name,
            model=args.model,
            instructions=instr,
            vector_store_id=vs_id,
            assistant_id=asst_id,
            update_existing=args.update_existing,
        )

    # --- save state
    state.update(
        {
            "vector_store_id": vs_id,
            "assistant_id": asst_id,
            "model": args.model,
            "root": str(pathlib.Path(args.root).resolve()),
            "include": args.include,
            "exclude": args.exclude,
        }
    )
    if not args.dry_run:
        save_state(state)
        print(f"[OK] State salvat în: {STATE_PATH}")

    print("\n=== Rezumat ===")
    print(f"Vector Store ID : {vs_id}")
    print(f"Assistant ID    : {asst_id}")
    print(f"Model           : {args.model}")
    print("Gata. 🚀")


if __name__ == "__main__":
    main()
