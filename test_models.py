"""
test_models.py
===============
Utility script to verify which OpenRouter models are accessible on your account.
Run this BEFORE running the benchmark to make sure all 5 configured models work.

Usage:
    python test_models.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from config import MODELS

load_dotenv()


def test_models():
    """Test each model in config.MODELS with a simple text prompt."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("Testing models configured in config.MODELS...")
    print("=" * 70)

    working = []
    for key, model_id in MODELS.items():
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with only: OK"}],
                max_tokens=10,
            )
            content = response.choices[0].message.content.strip()
            print(f"[PASS] {key:<15} -> {model_id}")
            print(f"       Response: {content[:50]}")
            working.append(key)
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"[FAIL] {key:<15} -> {model_id}")
            print(f"       Error: {error_msg}")

    print("=" * 70)
    print(f"Working: {len(working)}/{len(MODELS)} models")
    if len(working) == len(MODELS):
        print("All models accessible. You can run the full benchmark.")
    else:
        print("Some models failed. Check your OpenRouter account credits and access.")


if __name__ == "__main__":
    test_models()
