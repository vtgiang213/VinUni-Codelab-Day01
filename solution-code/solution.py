"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Student solution implementing the model-calling helpers and comparison utilities.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.6-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.6-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            text = _coerce_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if hasattr(value, "text"):
        return _coerce_text(getattr(value, "text"))
    return str(value)


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    text = ""
    if response and getattr(response, "choices", None):
        text = _coerce_text(response.choices[0].message.content)

    usage = {
        "input_tokens": int(response.usage.prompt_tokens),
        "output_tokens": int(response.usage.completion_tokens),
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.6 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.perf_counter() - start

    text = _coerce_text(getattr(response, "text", ""))
    usage = {
        "input_tokens": int(response.usage_metadata.prompt_token_count),
        "output_tokens": int(response.usage_metadata.candidates_token_count),
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start

    content = getattr(response, "content", [])
    text = ""
    if isinstance(content, list):
        for block in content:
            if hasattr(block, "text"):
                text += _coerce_text(block.text)
    else:
        text = _coerce_text(content)

    usage = {
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 3.6 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    responses = {}
    for key, label, call in (
        ("gpt4o", "OpenAI GPT-4o", lambda: call_openai(prompt, OPENAI_MODEL)),
        (
            "gpt4o_mini",
            "OpenAI GPT-4o Mini",
            lambda: call_openai(prompt, OPENAI_MINI_MODEL),
        ),
        ("gemini_flash", "Gemini Flash", lambda: call_gemini(prompt, GEMINI_MODEL)),
    ):
        print(f"Waiting for {label}...", flush=True)
        responses[key] = call()

    result: dict[str, dict[str, Any]] = {}
    for key, (text, latency, usage) in responses.items():
        model_name = {
            "gpt4o": "gpt-4o",
            "gpt4o_mini": "gpt-4o-mini",
            "gemini_flash": "gemini-3.6-flash",
        }[key]
        rate = PRICING_1M_TOKENS[model_name]
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
        cost = (
            (input_tokens * rate["input"]) + (output_tokens * rate["output"])
        ) / 1_000_000.0
        result[key] = {
            "response": text,
            "latency": float(latency),
            "cost": float(cost),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return result


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 3.6 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[dict[str, Any]] = []
    print("=== Gemini 3.6 Chatbot ===")
    print("Type 'quit' or 'exit' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            return

        history.append({"role": "user", "parts": [user_input]})
        history = history[-6:]

        print("Bot: ", end="", flush=True)
        reply_parts: list[str] = []
        try:
            stream = client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=history,
            )
            for chunk in stream:
                chunk_text = _coerce_text(getattr(chunk, "text", ""))
                if chunk_text:
                    print(chunk_text, end="", flush=True)
                    reply_parts.append(chunk_text)
            print()
        except Exception as exc:  # pragma: no cover - safeguard for interactive usage
            print(f"\n[streaming error] {exc}")
            return

        history.append({"role": "model", "parts": ["".join(reply_parts)]})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: PERF203 - explicit retry control is required
            last_error = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_with_backoff exhausted without a result")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results: list[dict] = []
    for prompt in prompts:
        try:
            result = compare_models(prompt)
        except TypeError:
            result = compare_models()
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]

    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    for item in results:
        prompt = str(item.get("prompt", ""))
        for model_key, stats in item.items():
            if model_key == "prompt":
                continue
            response = str(stats.get("response", "")).replace("\n", " ").strip()
            if len(response) > 50:
                response = response[:47] + "..."
            latency = float(stats.get("latency", 0.0))
            input_tokens = int(stats.get("input_tokens", 0))
            output_tokens = int(stats.get("output_tokens", 0))
            cost = float(stats.get("cost", 0.0))
            lines.append(
                f"| {prompt} | {model_names.get(model_key, model_key)} | {response} | {latency:.2f}s | {input_tokens}/{output_tokens} | ${cost:.6f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===", flush=True)
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===", flush=True)
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")