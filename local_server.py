"""Run the whole chatbot locally: UI + API on http://localhost:8000

Usage:
    python3 -m pip install -r requirements.txt uvicorn
    python3 local_server.py
"""
import os
from pathlib import Path

# load .env (GROQ_API_KEY) without needing python-dotenv
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import uvicorn

from api.index import app  # API routes + chat UI (public/ is mounted in api/index.py)

if __name__ == "__main__":
    key = os.environ.get("GROQ_API_KEY", "")
    print(f"GROQ_API_KEY: {'set (' + key[:8] + '…)' if key else 'MISSING — put it in .env'}")
    print("Chatbot running at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
