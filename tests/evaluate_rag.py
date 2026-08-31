"""
KnowRAG — Automated RAG Pipeline Evaluation & Regression Test Suite
------------------------------------------------------------------
This script runs automated evaluation tests against the KnowRAG RAG pipeline,
verifying factual correctness, source document attribution, and out-of-KB
safe refusal behavior.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 standard output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.rag_pipeline import ask_question, create_rag_pipeline


def run_evaluation() -> bool:
    """
    Execute the 6 evaluation test cases against the RAG pipeline.

    Returns:
        bool: True if all 6 tests PASS, False otherwise.
    """
    print("=" * 60)
    print("Starting KnowRAG Automated RAG Pipeline Evaluation")
    print("=" * 60)

    # Initialize RAG components once
    vector_index, groq_llm = create_rag_pipeline(temperature=0.0)

    test_cases: List[Dict[str, Any]] = [
        {
            "id": 1,
            "question": "What services does the university library provide?",
            "expected_sources": ["library_services.txt"],
            "is_out_of_kb": False,
            "description": "Library services grounding check",
        },
        {
            "id": 2,
            "question": "What academic programs are offered by the university?",
            "expected_sources": ["academic_programs.txt"],
            "is_out_of_kb": False,
            "description": "Academic degree programs check",
        },
        {
            "id": 3,
            "question": "What rules should students follow on campus?",
            "expected_sources": ["campus_rules.txt"],
            "is_out_of_kb": False,
            "description": "Campus rules and student conduct check",
        },
        {
            "id": 4,
            "question": "What support is available to students?",
            "expected_sources": ["student_services.txt", "library_services.txt"],
            "is_out_of_kb": False,
            "description": "Student support services check",
        },
        {
            "id": 5,
            "question": "What is TechNova University?",
            "expected_sources": ["university_overview.txt", "academic_programs.txt"],
            "is_out_of_kb": False,
            "description": "University overview and mission check",
        },
        {
            "id": 6,
            "question": "What is the university's policy on underwater basket weaving?",
            "expected_sources": [],
            "is_out_of_kb": True,
            "description": "Out-of-knowledge-base safe refusal check",
        },
    ]

    results = []

    for tc in test_cases:
        t_id = tc["id"]
        question = tc["question"]
        expected_sources = tc["expected_sources"]
        is_out_of_kb = tc["is_out_of_kb"]

        print(f"\nEvaluating TEST {t_id}: {question}")

        # Execute with retry on transient rate limits
        max_retries = 5
        res = None
        for attempt in range(max_retries):
            try:
                res = ask_question(question=question, index=vector_index, llm=groq_llm)
                break
            except Exception as ex:
                if ("429" in str(ex) or "RateLimitError" in type(ex).__name__) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  [Notice] Rate limit hit. Backing off {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise ex

        try:
            answer = res["answer"].strip()
            sources = res["sources"]

            # Evaluation criteria checks
            test_passed = True
            failure_reasons = []

            if is_out_of_kb:
                # Test 6: Must refuse safely
                refusal_phrases = [
                    "not available in the knowledge base",
                    "information is not available",
                    "not mentioned in the provided context",
                ]
                has_refusal = any(p in answer.lower() for p in refusal_phrases)
                if not has_refusal:
                    test_passed = False
                    failure_reasons.append("Answer did not safely state that information is unavailable.")
                if sources:
                    test_passed = False
                    failure_reasons.append(f"Expected 0 sources for out-of-KB query, got: {sources}")
            else:
                # Tests 1-5: Must have non-empty grounded answer and match expected sources
                if not answer or len(answer) < 20:
                    test_passed = False
                    failure_reasons.append("Answer was unexpectedly empty or too short.")
                if "not available in the knowledge base" in answer.lower():
                    test_passed = False
                    failure_reasons.append("Valid in-KB question was incorrectly refused.")
                
                # Check if at least one expected source is present in returned sources
                matched_source = any(es in sources for es in expected_sources)
                if not matched_source:
                    test_passed = False
                    failure_reasons.append(f"Expected source from {expected_sources}, but got {sources}")

            status_str = "PASS" if test_passed else "FAIL"
            results.append((t_id, status_str, failure_reasons))

            print(f"Answer: {answer[:120]}...")
            print(f"Sources: {sources}")
            print(f"Result: {status_str}")
            if not test_passed:
                for r in failure_reasons:
                    print(f"  [Failure reason]: {r}")

        except Exception as e:
            print(f"Result: FAIL (Exception: {e})")
            results.append((t_id, "FAIL", [str(e)]))

        # Small delay between questions to respect API pacing
        time.sleep(2)

    # Print Formatted Summary
    print("\n" + "=" * 60)
    print("KnowRAG RAG Evaluation")
    print("=" * 60)

    pass_count = 0
    for t_id, status_str, _ in results:
        print(f"TEST {t_id}: {status_str}")
        if status_str == "PASS":
            pass_count += 1

    total_count = len(results)
    print(f"\nTotal: {pass_count}/{total_count} PASS")
    print("=" * 60)

    return pass_count == total_count


if __name__ == "__main__":
    all_passed = run_evaluation()
    sys.exit(0 if all_passed else 1)
