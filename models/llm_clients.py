"""
LLM client functions for Groq and HuggingFace APIs.
Gemini has been removed.
"""

import time
import requests

# ── Constants ────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.1-8b-instant"
HF_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT_TEMPLATES = {
    "zero_shot": (
        "You are a knowledgeable finance assistant. "
        "Answer the following question concisely and accurately.\n\n"
        "Question: {question}\nAnswer:"
    ),
    "few_shot": (
        "You are a knowledgeable finance assistant. "
        "Here are two example finance Q&A pairs:\n\n"
        "Q: What is compound interest?\n"
        "A: Compound interest is interest calculated on both the initial principal "
        "and the accumulated interest from previous periods.\n\n"
        "Q: What is a mutual fund?\n"
        "A: A mutual fund pools money from many investors to purchase a diversified "
        "portfolio of stocks, bonds, or other securities.\n\n"
        "Now answer:\nQ: {question}\nA:"
    ),
    "chain_of_thought": (
        "You are a knowledgeable finance assistant. "
        "Think through the question step by step before giving your final answer.\n\n"
        "Question: {question}\n\n"
        "Step-by-step reasoning:\n1."
    ),
}


# ── Groq ─────────────────────────────────────────────────────────────────────

def get_groq_response(
    question: str,
    api_key: str,
    prompt_style: str = "zero_shot",
) -> dict:
    """
    Query the Groq API with mixtral-8x7b-32768.

    Returns:
        dict with keys: answer, model, prompt_style, response_time, error
    """
    result = {
        "answer": "",
        "model": GROQ_MODEL,
        "prompt_style": prompt_style,
        "response_time": 0.0,
        "error": None,
    }

    prompt = PROMPT_TEMPLATES.get(prompt_style, PROMPT_TEMPLATES["zero_shot"]).format(
        question=question
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        start = time.time()
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        elapsed = round(time.time() - start, 3)
        result["response_time"] = elapsed

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        data = resp.json()
        result["answer"] = data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out after 30 s."
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── HuggingFace ───────────────────────────────────────────────────────────────

from huggingface_hub import InferenceClient
import time


def get_huggingface_response(
    question: str,
    api_key: str,
    prompt_style: str = "zero_shot",
) -> dict:
    """
    Query the HuggingFace Inference API with Mistral-7B-Instruct-v0.2.

    Returns:
        dict with keys: answer, model, prompt_style, response_time, error
    """
    result = {
        "answer": "",
        "model": HF_MODEL,
        "prompt_style": prompt_style,
        "response_time": 0.0,
        "error": None,
    }

    prompt = PROMPT_TEMPLATES.get(
        prompt_style, PROMPT_TEMPLATES["zero_shot"]
    ).format(question=question)

    try:
        client = InferenceClient(token=api_key)

        start = time.time()

        response = client.chat_completion(
            model=HF_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2,
        )

        elapsed = round(time.time() - start, 3)
        result["response_time"] = elapsed

        result["answer"] = response.choices[0].message["content"].strip()

    except Exception as exc:
        result["error"] = str(exc)

    return result