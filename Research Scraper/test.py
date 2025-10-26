from pathlib import Path
import os

KEY_PATH = Path("secrets/gpt_secret.txt")

if not KEY_PATH.is_file():
    raise FileNotFoundError(f"Missing {KEY_PATH}. Put your key in that file.")

api_key = KEY_PATH.read_text(encoding="utf-8").strip()  # trim newline/whitespace
if not api_key or len(api_key) < 20:
    raise ValueError("OPENAI_API_KEY looks empty or invalid.")

os.environ["OPENAI_API_KEY"] = api_key  # set for current process (and children)

from openai import OpenAI

client = OpenAI()  # auto-reads OPENAI_API_KEY from os.environ
resp = client.responses.create(model="gpt-4o-mini", input="Say OK.", temperature=0)
print(resp.output_text)
