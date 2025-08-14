# scripts/test_assistant_query.py
import os
import sys
import time
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()  # încarcă variabilele din .env în os.environ

QUESTION = "Explică pe scurt ce face funcția batch_process din app/stock_analysis.py și arată-mi pașii principali."


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    asst_id = os.environ.get("ASSISTANT_ID")

    if not api_key:
        print("[FATAL] OPENAI_API_KEY lipsă în environment.")
        sys.exit(1)
    if not asst_id:
        print("[FATAL] ASSISTANT_ID lipsă în environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # poți trece întrebarea din linia de comandă: python scripts/test_assistant_query.py "întrebarea ta"
    question = " ".join(sys.argv[1:]).strip() or QUESTION

    # 1) creăm un thread și adăugăm mesajul user
    thread = client.beta.threads.create(messages=[{"role": "user", "content": question}])

    # 2) rulăm asistentul pe thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=asst_id,
        # dacă ai atașat Vector Store la asistent în bootstrap, nu mai trebuie să-l trimiți aici
        # tool_resources={"file_search": {"vector_store_ids": [os.environ.get("VECTOR_STORE_ID")]}}
    )

    # 3) poll până se termină
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in ("completed", "failed", "cancelled", "expired", "requires_action"):
            break
        time.sleep(0.8)

    if run.status != "completed":
        print(f"[ERR] Run status = {run.status}")
        if getattr(run, "last_error", None):
            print("       last_error:", run.last_error)
        sys.exit(2)

    # 4) citim răspunsurile asistentului
    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    print("\n===== ASSISTANT =====")
    for m in msgs.data:
        if m.role == "assistant":
            for c in m.content:
                if getattr(c, "type", None) == "text":
                    print(c.text.value)
    print("=====================\n")


if __name__ == "__main__":
    main()
