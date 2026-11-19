import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv(override=True)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60000),  # 60 seconds, in milliseconds
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "").split(",") if m.strip()]
MODELS = [MODEL] + FALLBACKS

COOLDOWN_SECONDS = 300   # skip a busy model for 5 minutes
cooldown = {}            # model name -> time when we may try it again
LAST_MODEL = None        # the model that answered most recently


def generate(contents, config):
    """Try models in order, skipping ones that were recently busy."""
    global LAST_MODEL
    last_error = None
    now = time.time()
    candidates = [m for m in MODELS if cooldown.get(m, 0) <= now] or MODELS

    for model in candidates:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                LAST_MODEL = model
                return resp
            except errors.APIError as e:
                last_error = e
                if e.code == 404:  # model not available to this key
                    cooldown[model] = time.time() + 3600
                    break
                if e.code in (429, 500, 503, 504):
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    cooldown[model] = time.time() + COOLDOWN_SECONDS
                    break
                raise
            except Exception as e:  # network hiccups and client-side timeouts
                last_error = e
                time.sleep(2)
        print(f"  (Skipping {model}, trying the next model...)", flush=True)
    raise last_error


def ask(system, user, json_mode=False):
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.2,
        response_mime_type="application/json" if json_mode else None,
    )
    resp = generate(user, config)
    return resp.text or ""


def ask_json(system, user):
    text = ask(system, user, json_mode=True).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)