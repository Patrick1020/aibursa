# .ci/sync_to_openai.py
import os
import glob
from openai import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

paths = [
    p
    for p in glob.glob("**/*", recursive=True)
    if p.endswith(
        (
            ".py",
            ".md",
            ".sql",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".csv",
            ".html",
            ".ts",
            ".tsx",
            ".css",
        )
    )
    and (".venv" not in p and ".git" not in p and "models_store" not in p and "static" not in p)
]
files = [open(p, "rb") for p in paths]
res = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=VECTOR_STORE_ID, files=files
)
print(f"[SYNC] uploaded={len(paths)} status={res.status}")
