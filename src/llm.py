"""
KnowRAG - LLM Integration Module
---------------------------------
This module provides the LLM (Large Language Model) integration for KnowRAG
using LlamaIndex's Groq integration (llama_index.llms.groq.Groq).

Features:
- Secure API key retrieval via python-dotenv / environment variables.
- Configurable models available on Groq (default: openai/gpt-oss-20b).
- Isolated LLM factory function (create_llm) and connectivity test.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 standard output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import find_dotenv, load_dotenv
from llama_index.llms.groq import Groq

# Supported default model available on Groq
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.1


def load_environment() -> bool:
    """
    Load environment variables from .env file securely.
    Checks project root .env, .venv/.env, and standard dotenv search paths.

    Returns:
        bool: True if GROQ_API_KEY is found in the environment, False otherwise.
    """
    # Attempt loading from root .env first
    root_env = PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=True)

    # Attempt loading from .venv/.env if not already loaded
    venv_env = PROJECT_ROOT / ".venv" / ".env"
    if venv_env.exists() and not os.getenv("GROQ_API_KEY"):
        load_dotenv(dotenv_path=venv_env, override=True)

    # Fallback to find_dotenv
    if not os.getenv("GROQ_API_KEY"):
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(dotenv_path=found, override=True)

    api_key = os.getenv("GROQ_API_KEY")
    return bool(api_key and len(api_key.strip()) > 0)


def create_llm(
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
) -> Groq:
    """
    Create and return a configured LlamaIndex Groq LLM instance.

    Args:
        model (Optional[str]): The model identifier on Groq. If not provided,
            checks GROQ_MODEL environment variable, then defaults to 'openai/gpt-oss-20b'.
        temperature (float): Sampling temperature (0.0 to 1.0). Default is 0.1.
        **kwargs: Additional keyword arguments forwarded to the Groq constructor.

    Returns:
        Groq: Initialized LlamaIndex Groq LLM instance.

    Raises:
        ValueError: If GROQ_API_KEY is missing or empty.
    """
    # Ensure environment is loaded
    is_key_present = load_environment()
    if not is_key_present:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please add 'GROQ_API_KEY=gsk_...' to your .env file."
        )

    api_key = os.getenv("GROQ_API_KEY")
    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    llm = Groq(
        model=selected_model,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
    return llm


def test_llm_generation(
    model: Optional[str] = None,
    test_prompt: str = "Hello! Please confirm in one short sentence that you are operational.",
) -> bool:
    """
    Perform an isolated test of the Groq LLM integration.

    Args:
        model (Optional[str]): Optional model name to test.
        test_prompt (str): Prompt to send for testing generation.

    Returns:
        bool: True if test succeeded, False otherwise.
    """
    print("=" * 70)
    print("KnowRAG: Isolated Groq LLM Integration Test")
    print("=" * 70)

    # 1. Verify API Key detection (without printing key value)
    key_detected = load_environment()
    if key_detected:
        print("[OK] GROQ_API_KEY detected in environment (value hidden for security).")
    else:
        print("[ERROR] GROQ_API_KEY not found in .env or environment variables.")
        print("        Please ensure GROQ_API_KEY is defined in .env")
        return False

    target_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    print(f"[*] Target Model: {target_model}")

    # 2. Initialize LLM
    try:
        llm = create_llm(model=target_model)
        print(f"[OK] Groq LLM initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq LLM: {e}")
        return False

    # 3. Test prompt generation
    print(f"[*] Sending test prompt: \"{test_prompt}\"")
    try:
        response = llm.complete(test_prompt)
        print("\n--- LLM Response ---")
        print(response.text.strip())
        print("--------------------")
        print("\n[SUCCESS] Test prompt received a valid response from Groq LLM.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        # If model is deprecated or invalid, suggest fallback
        if "model_not_found" in str(e).lower() or "decommissioned" in str(e).lower():
            print("\n[TIP] Model might be deprecated. Supported models on Groq include:")
            print("      - llama-3.3-70b-versatile")
            print("      - llama-3.1-8b-instant")
            print("      - llama3-70b-8192")
            print("      - llama3-8b-8192")
        return False


if __name__ == "__main__":
    success = test_llm_generation()
    if not success:
        sys.exit(1)
