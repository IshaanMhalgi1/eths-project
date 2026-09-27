import os
import sys
import requests

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("LLM_API_KEY", "")
base = os.environ.get("LLM_BASE_URL", "").rstrip("/")

if not key:
    print("no key")
    sys.exit(1)

r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=60)
print("GET /models ->", r.status_code)
if r.status_code != 200:
    print(r.text[:400])
    sys.exit(1)

data = r.json().get("data", [])
models = sorted(m.get("id", "") for m in data)
print(f"\n{len(models)} models available:\n")
for m in models:
    print(" ", m)
