import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
# Get API token from environment variable
api_key = os.getenv('GEMINI_TOKEN')

def improve_agent_details(text: str) -> str:
    """
    Enhance a given text to make it more suitable for question answering using the SQuAD2 model.
    Returns the improved version as plain text.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")

    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    )

    prompt = (
        "Improve the following passage to be more informative and detailed. "
        "The goal is to optimize it for use in a question answering system based on the deepset/roberta-base-squad2 model.\n\n"
        "Original text:\n" + text
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        candidates = result.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini API.")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or not parts[0].get("text"):
            raise RuntimeError("No valid content returned from Gemini.")

        return parts[0]["text"].strip()

    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}")


