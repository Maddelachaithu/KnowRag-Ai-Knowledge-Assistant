"""
KnowRAG — Comprehensive Conversational & Functional Test Suite
--------------------------------------------------------------
Automated test suite verifying the 17 specified test cases:
 1. Greeting detection
 2. "hello" response
 3. "hi" response
 4. /help command
 5. /show documents command
 6. /clear command
 7. /about command
 8. "thanks" response
 9. "bye" response
10. Brainstorming/general question
11. Existing RAG question
12. Out-of-knowledge-base question
13. PDF upload and query
14. TXT upload and query
15. Duplicate upload protection
16. Source citation verification
17. Missing GROQ_API_KEY handling

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import io
import os
import sys
import time
from pathlib import Path

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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.conversation import (
    COMMAND_ABOUT,
    COMMAND_CLEAR,
    COMMAND_HELP,
    COMMAND_SHOW_DOCS,
    detect_command,
    detect_greeting,
    process_user_query,
)
from src.ingestion import process_uploaded_file
from src.llm import create_llm
from src.rag_pipeline import create_rag_pipeline
from src.vector_store import clear_uploaded_documents, index_uploaded_document


def create_sample_pdf_bytes() -> bytes:
    """Generate in-memory sample 2-page PDF document."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(72, 750, "TechNova AI Research Laboratory Guidelines")
    c.drawString(72, 720, "The AI Lab is located on the 4th floor of the Turing Building.")
    c.drawString(72, 700, "High-Performance GPU clusters require faculty advisor pre-approval.")
    c.showPage()
    c.drawString(72, 750, "AI Lab Access Hours")
    c.drawString(72, 720, "Graduate researchers have 24/7 keycard access to the server room.")
    c.save()
    return buffer.getvalue()


def create_sample_txt_bytes() -> bytes:
    """Generate in-memory sample TXT document."""
    content = (
        "TechNova Drone Flight Policy\n"
        "Unmanned Aerial Vehicles (drones) may only be operated on the East Athletic Field.\n"
        "Maximum flight altitude on campus is restricted to 120 feet above ground level.\n"
        "Flights during varsity sporting events are strictly prohibited.\n"
    )
    return content.encode("utf-8")


def run_all_tests() -> bool:
    print("=" * 70)
    print("Starting KnowRAG Comprehensive Conversational & Functional Test Suite")
    print("=" * 70)

    # Initialize RAG Pipeline
    vector_index, groq_llm = create_rag_pipeline(temperature=0.0)
    clear_uploaded_documents(index=vector_index)

    results = []

    def log_result(test_num: int, name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        results.append((test_num, name, status, details))
        print(f"TEST {test_num:02d}: [{status}] {name}")
        if details:
            print(f"         {details}")

    # -------------------------------------------------------------
    # 1. Greeting detection
    # -------------------------------------------------------------
    g1 = detect_greeting("hello")
    g2 = detect_greeting("good morning!")
    g3 = detect_greeting("hey there")
    g4 = detect_greeting("what is technova?")  # Should NOT be a greeting
    pass_1 = bool(g1 and g2 and g3 and not g4)
    log_result(1, "Greeting Detection", pass_1, f"g1={g1}, g2={g2}, g3={g3}, non-greeting={g4}")

    # -------------------------------------------------------------
    # 2. "hello" response
    # -------------------------------------------------------------
    res_hello = process_user_query("hello", index=vector_index, llm=groq_llm)
    pass_2 = res_hello.get("intent") == "greeting" and "How can I help" in res_hello.get("answer", "")
    log_result(2, "\"hello\" Response", pass_2, f"Answer: {res_hello.get('answer')[:60]}...")

    # -------------------------------------------------------------
    # 3. "hi" response
    # -------------------------------------------------------------
    res_hi = process_user_query("Hi", index=vector_index, llm=groq_llm)
    pass_3 = res_hi.get("intent") == "greeting" and "KnowRAG" in res_hi.get("answer", "")
    log_result(3, "\"hi\" Response", pass_3, f"Answer: {res_hi.get('answer')[:60]}...")

    # -------------------------------------------------------------
    # 4. /help command
    # -------------------------------------------------------------
    res_help = process_user_query("/help", index=vector_index, llm=groq_llm)
    pass_4 = (
        res_help.get("intent") == "command"
        and "/show documents" in res_help.get("answer", "")
        and "/clear" in res_help.get("answer", "")
    )
    log_result(4, "/help Command", pass_4, f"Contains commands guide: {pass_4}")

    # -------------------------------------------------------------
    # 5. /show documents command
    # -------------------------------------------------------------
    res_docs = process_user_query("/show documents", index=vector_index, llm=groq_llm)
    pass_5 = (
        res_docs.get("intent") == "command"
        and "academic_programs.txt" in res_docs.get("answer", "")
        and "library_services.txt" in res_docs.get("answer", "")
    )
    log_result(5, "/show documents Command", pass_5, f"Lists default documents: {pass_5}")

    # -------------------------------------------------------------
    # 6. /clear command
    # -------------------------------------------------------------
    res_clear = process_user_query("/clear", index=vector_index, llm=groq_llm)
    pass_6 = (
        res_clear.get("intent") == "command"
        and res_clear.get("action") == "clear_chat"
        and "cleared" in res_clear.get("answer", "").lower()
    )
    log_result(6, "/clear Command", pass_6, f"Action: {res_clear.get('action')}, Answer: {res_clear.get('answer')}")

    # -------------------------------------------------------------
    # 7. /about command
    # -------------------------------------------------------------
    res_about = process_user_query("/about", index=vector_index, llm=groq_llm)
    pass_7 = (
        res_about.get("intent") == "command"
        and "Retrieval-Augmented Generation" in res_about.get("answer", "")
    )
    log_result(7, "/about Command", pass_7, f"Contains RAG description: {pass_7}")

    # -------------------------------------------------------------
    # 8. "thanks" response
    # -------------------------------------------------------------
    res_thanks = process_user_query("Thank you so much!", index=vector_index, llm=groq_llm)
    pass_8 = res_thanks.get("intent") == "casual" and "welcome" in res_thanks.get("answer", "").lower()
    log_result(8, "\"thanks\" Response", pass_8, f"Answer: {res_thanks.get('answer')}")

    # -------------------------------------------------------------
    # 9. "bye" response
    # -------------------------------------------------------------
    res_bye = process_user_query("Goodbye", index=vector_index, llm=groq_llm)
    pass_9 = res_bye.get("intent") == "casual" and "goodbye" in res_bye.get("answer", "").lower()
    log_result(9, "\"bye\" Response", pass_9, f"Answer: {res_bye.get('answer')}")

    time.sleep(1)

    # -------------------------------------------------------------
    # 10. Brainstorming/general question
    # -------------------------------------------------------------
    res_brainstorm = process_user_query("give me 3 cybersecurity project ideas", index=vector_index, llm=groq_llm)
    pass_10 = (
        res_brainstorm.get("intent") == "brainstorming"
        and len(res_brainstorm.get("answer", "")) > 50
        and len(res_brainstorm.get("sources", [])) == 0  # No fabricated sources
    )
    log_result(10, "Brainstorming / General Question", pass_10, f"Intent: {res_brainstorm.get('intent')}, Sources: {res_brainstorm.get('sources')}")

    time.sleep(2)

    # -------------------------------------------------------------
    # 11. Existing RAG question
    # -------------------------------------------------------------
    res_rag = process_user_query("What academic programs are offered by the university?", index=vector_index, llm=groq_llm)
    pass_11 = (
        res_rag.get("intent") == "rag"
        and "academic_programs.txt" in res_rag.get("sources", [])
        and len(res_rag.get("answer", "")) > 30
    )
    log_result(11, "Existing RAG Question", pass_11, f"Sources: {res_rag.get('sources')}")

    time.sleep(2)

    # -------------------------------------------------------------
    # 12. Out-of-knowledge-base question
    # -------------------------------------------------------------
    res_oob = process_user_query("What is the university's policy on underwater basket weaving?", index=vector_index, llm=groq_llm)
    pass_12 = (
        res_oob.get("intent") == "rag"
        and "not available in the knowledge base" in res_oob.get("answer", "").lower()
        and len(res_oob.get("sources", [])) == 0
    )
    log_result(12, "Out-of-Knowledge-Base Refusal", pass_12, f"Answer: {res_oob.get('answer')[:70]}...")

    time.sleep(2)

    # -------------------------------------------------------------
    # 13. PDF upload and query
    # -------------------------------------------------------------
    pdf_fn = "ai_lab_guidelines.pdf"
    pdf_bytes = create_sample_pdf_bytes()
    pdf_docs, _ = process_uploaded_file(pdf_fn, pdf_bytes, save_to_disk=True)
    idx_pdf_res = index_uploaded_document(pdf_docs, index=vector_index)

    res_pdf_q = process_user_query("Where is the AI Research Lab located and on what floor?", index=vector_index, llm=groq_llm)
    pass_13 = (
        idx_pdf_res.get("status") == "success"
        and pdf_fn in res_pdf_q.get("sources", [])
        and "turing" in res_pdf_q.get("answer", "").lower()
    )
    log_result(13, "PDF Upload & Query", pass_13, f"Sources: {res_pdf_q.get('sources')}")

    time.sleep(2)

    # -------------------------------------------------------------
    # 14. TXT upload and query
    # -------------------------------------------------------------
    txt_fn = "drone_flight_policy.txt"
    txt_bytes = create_sample_txt_bytes()
    txt_docs, _ = process_uploaded_file(txt_fn, txt_bytes, save_to_disk=True)
    idx_txt_res = index_uploaded_document(txt_docs, index=vector_index)

    res_txt_q = process_user_query("What is the maximum flight altitude for drones on campus?", index=vector_index, llm=groq_llm)
    pass_14 = (
        idx_txt_res.get("status") == "success"
        and txt_fn in res_txt_q.get("sources", [])
        and "120" in res_txt_q.get("answer", "")
    )
    log_result(14, "TXT Upload & Query", pass_14, f"Sources: {res_txt_q.get('sources')}")

    time.sleep(2)

    # -------------------------------------------------------------
    # 15. Duplicate upload protection
    # -------------------------------------------------------------
    dup_docs, _ = process_uploaded_file(pdf_fn, pdf_bytes, save_to_disk=False)
    dup_res = index_uploaded_document(dup_docs, index=vector_index)
    pass_15 = dup_res.get("status") == "duplicate" and dup_res.get("chunks_created") == 0
    log_result(15, "Duplicate Upload Protection", pass_15, f"Status: {dup_res.get('status')}, Chunks: {dup_res.get('chunks_created')}")

    # -------------------------------------------------------------
    # 16. Source citation verification
    # -------------------------------------------------------------
    details = res_pdf_q.get("source_details", [])
    has_page_citation = any("Page" in d for d in details) or pdf_fn in res_pdf_q.get("sources", [])
    pass_16 = has_page_citation and len(res_pdf_q.get("sources", [])) > 0
    log_result(16, "Source Citation Verification", pass_16, f"Source Details: {details}")

    # Clean up uploaded test chunks
    clear_uploaded_documents(index=vector_index)

    # -------------------------------------------------------------
    # 17. Missing GROQ_API_KEY handling
    # -------------------------------------------------------------
    missing_key_handled = False
    try:
        create_llm(api_key="")
    except ValueError as ex:
        if "GROQ_API_KEY" in str(ex):
            missing_key_handled = True
    pass_17 = missing_key_handled
    log_result(17, "Missing GROQ_API_KEY Handling", pass_17, f"Gracefully caught missing API key: {pass_17}")

    # Print Formatted Summary
    print("\n" + "=" * 70)
    print("KnowRAG 17-Point Conversational & Functional Test Suite Summary")
    print("=" * 70)

    pass_count = sum(1 for _, _, status, _ in results if status == "PASS")
    total_count = len(results)

    for num, name, status, _ in results:
        print(f"TEST {num:02d}: {status} - {name}")

    print(f"\nTotal: {pass_count}/{total_count} PASS")
    print("=" * 70)

    return pass_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
