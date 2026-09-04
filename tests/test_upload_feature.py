"""
KnowRAG — Automated Upload Feature Test Suite
---------------------------------------------
This test suite verifies the dynamic document upload pipeline:
- TEST A: Upload a TXT document and verify retrieval.
- TEST B: Upload a PDF document and verify its page-aware content retrieval.
- TEST C: Ask a question about the uploaded document and verify grounded LLM answer + source citation.
- TEST D: Upload the same document twice and verify duplicate chunks are rejected.
- TEST E: Ask an unrelated question and verify safe refusal (no hallucination).
- TEST F: Verify the original five documents still work concurrently.

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import io
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

from src.ingestion import process_uploaded_file
from src.rag_pipeline import ask_question, create_rag_pipeline
from src.retrieval import retrieve_documents
from src.vector_store import (
    clear_uploaded_documents,
    get_knowledge_base_stats,
    index_uploaded_document,
)


def create_sample_pdf_bytes() -> bytes:
    """Generate in-memory sample 2-page PDF document."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Page 1
    c.drawString(72, 750, "TechNova Shuttle Transportation Services")
    c.drawString(72, 720, "TechNova operates an eco-friendly electric campus shuttle fleet.")
    c.drawString(72, 700, "Shuttles run continuously from 6:00 AM to 11:30 PM at 15-minute intervals.")
    c.drawString(72, 680, "The North Gate Express connects the engineering complex to the central library.")
    c.showPage()

    # Page 2
    c.drawString(72, 750, "Campus Parking and Permit Guidelines")
    c.drawString(72, 720, "Students must register vehicles with Campus Safety before parking on university property.")
    c.drawString(72, 700, "Semester parking permits for Lot C cost 50 dollars.")
    c.drawString(72, 680, "Unauthorized vehicles parked in staff bays will be ticketed and towed.")
    c.save()

    return buffer.getvalue()


def create_sample_txt_bytes() -> bytes:
    """Generate in-memory sample TXT document."""
    content = (
        "TechNova Innovation Lab Guidelines\n"
        "The TechNova Innovation Lab is accessible 24/7 for students enrolled in research projects.\n"
        "Safety training certification Level 2 is mandatory prior to using 3D printers and laser cutters.\n"
        "Reservations for prototyping benches can be made through the student portal.\n"
    )
    return content.encode("utf-8")


def run_upload_tests() -> bool:
    """Run tests A through F for the upload feature."""
    print("=" * 65)
    print("Starting KnowRAG Dynamic Document Upload Feature Tests")
    print("=" * 65)

    # Step 0: Ensure baseline pipeline is ready and clean any prior test uploads
    vector_index, groq_llm = create_rag_pipeline(temperature=0.0)
    clear_uploaded_documents(index=vector_index)

    initial_stats = get_knowledge_base_stats()
    print(f"Initial KB: {initial_stats['total_documents']} docs, {initial_stats['total_chunks']} chunks")
    print("=" * 65)

    all_passed = True
    test_results = []

    # -------------------------------------------------------------
    # TEST A: Upload a TXT document and verify retrieval
    # -------------------------------------------------------------
    print("\n--- TEST A: Ingest & Retrieve TXT Document ---")
    txt_filename = "innovation_lab_guidelines.txt"
    txt_bytes = create_sample_txt_bytes()

    try:
        txt_docs, txt_doc_id = process_uploaded_file(txt_filename, txt_bytes, save_to_disk=True)
        res_txt = index_uploaded_document(documents=txt_docs, index=vector_index)
        print(f"Indexing result: {res_txt['status']} ({res_txt['chunks_created']} chunks)")

        # Verify retrieval
        retrieved = retrieve_documents("What safety certification is required for 3D printers in the Innovation Lab?", top_k=3, index=vector_index)
        found_txt = any(node.node.metadata.get("file_name") == txt_filename for node in retrieved)

        if res_txt["status"] == "success" and found_txt:
            print("TEST A Result: PASS (TXT document successfully indexed and retrieved)")
            test_results.append(("TEST A", "PASS"))
        else:
            print(f"TEST A Result: FAIL (Indexed: {res_txt['status']}, Retrieved: {found_txt})")
            test_results.append(("TEST A", "FAIL"))
            all_passed = False
    except Exception as e:
        print(f"TEST A Result: FAIL (Exception: {e})")
        test_results.append(("TEST A", "FAIL"))
        all_passed = False

    # -------------------------------------------------------------
    # TEST B: Upload a PDF document and verify page-aware retrieval
    # -------------------------------------------------------------
    print("\n--- TEST B: Ingest & Retrieve PDF Document with Page Metadata ---")
    pdf_filename = "transportation_guidelines.pdf"
    pdf_bytes = create_sample_pdf_bytes()

    try:
        pdf_docs, pdf_doc_id = process_uploaded_file(pdf_filename, pdf_bytes, save_to_disk=True)
        res_pdf = index_uploaded_document(documents=pdf_docs, index=vector_index)
        print(f"Indexing result: {res_pdf['status']} ({res_pdf['chunks_created']} chunks)")

        # Verify retrieval of page 1 and page 2 content
        retrieved_p1 = retrieve_documents("What hours does the electric campus shuttle operate?", top_k=3, index=vector_index)
        retrieved_p2 = retrieve_documents("How much is the semester parking permit for Lot C?", top_k=3, index=vector_index)

        p1_match = any(
            n.node.metadata.get("file_name") == pdf_filename and n.node.metadata.get("page_number") == 1
            for n in retrieved_p1
        )
        p2_match = any(
            n.node.metadata.get("file_name") == pdf_filename and n.node.metadata.get("page_number") == 2
            for n in retrieved_p2
        )

        if res_pdf["status"] == "success" and p1_match and p2_match:
            print("TEST B Result: PASS (PDF multi-page document indexed with page metadata and retrieved)")
            test_results.append(("TEST B", "PASS"))
        else:
            print(f"TEST B Result: FAIL (P1 Match: {p1_match}, P2 Match: {p2_match})")
            test_results.append(("TEST B", "FAIL"))
            all_passed = False
    except Exception as e:
        print(f"TEST B Result: FAIL (Exception: {e})")
        test_results.append(("TEST B", "FAIL"))
        all_passed = False

    time.sleep(2)

    # -------------------------------------------------------------
    # TEST C: Grounded LLM Q&A on Uploaded PDF
    # -------------------------------------------------------------
    print("\n--- TEST C: Grounded LLM Answer on Uploaded PDF ---")
    try:
        q_c = "What are the hours and interval for the TechNova electric campus shuttles?"
        qa_res = ask_question(question=q_c, index=vector_index, llm=groq_llm)

        ans = qa_res["answer"]
        sources = qa_res["sources"]
        source_details = qa_res["source_details"]

        has_hours = "6:00" in ans and "11:30" in ans
        has_source = pdf_filename in sources

        print(f"Answer: {ans[:140]}...")
        print(f"Sources: {sources}")
        print(f"Source Details: {source_details}")

        if has_hours and has_source:
            print("TEST C Result: PASS (Accurate grounded answer citing uploaded PDF)")
            test_results.append(("TEST C", "PASS"))
        else:
            print(f"TEST C Result: FAIL (Fact Check: {has_hours}, Source Check: {has_source})")
            test_results.append(("TEST C", "FAIL"))
            all_passed = False
    except Exception as e:
        print(f"TEST C Result: FAIL (Exception: {e})")
        test_results.append(("TEST C", "FAIL"))
        all_passed = False

    time.sleep(2)

    # -------------------------------------------------------------
    # TEST D: Duplicate Upload Protection
    # -------------------------------------------------------------
    print("\n--- TEST D: Duplicate Upload Protection ---")
    try:
        dup_docs, _ = process_uploaded_file(pdf_filename, pdf_bytes, save_to_disk=False)
        dup_res = index_uploaded_document(documents=dup_docs, index=vector_index)

        print(f"Duplicate upload attempt status: {dup_res['status']}")
        print(f"Message: {dup_res['message']}")

        if dup_res["status"] == "duplicate" and dup_res["chunks_created"] == 0:
            print("TEST D Result: PASS (Duplicate upload detected; 0 duplicate chunks created)")
            test_results.append(("TEST D", "PASS"))
        else:
            print(f"TEST D Result: FAIL (Expected duplicate status, got: {dup_res})")
            test_results.append(("TEST D", "FAIL"))
            all_passed = False
    except Exception as e:
        print(f"TEST D Result: FAIL (Exception: {e})")
        test_results.append(("TEST D", "FAIL"))
        all_passed = False

    time.sleep(2)

    # -------------------------------------------------------------
    # TEST E: Out-of-Scope Safe Refusal
    # -------------------------------------------------------------
    print("\n--- TEST E: Out-of-Scope Query Safe Refusal ---")
    try:
        q_e = "What is the policy for piloting hot air balloons on campus?"
        e_res = ask_question(question=q_e, index=vector_index, llm=groq_llm)

        e_ans = e_res["answer"]
        e_sources = e_res["sources"]

        refusal_phrases = [
            "not available in the knowledge base",
            "information is not available",
            "not mentioned in the provided context",
        ]
        has_refusal = any(p in e_ans.lower() for p in refusal_phrases)

        print(f"Answer: {e_ans}")
        print(f"Sources: {e_sources}")

        if has_refusal and len(e_sources) == 0:
            print("TEST E Result: PASS (Safely refused unsupported question without hallucination)")
            test_results.append(("TEST E", "PASS"))
        else:
            print(f"TEST E Result: FAIL (Refusal: {has_refusal}, Sources: {e_sources})")
            test_results.append(("TEST E", "FAIL"))
            all_passed = False
    except Exception as e:
        print(f"TEST E Result: FAIL (Exception: {e})")
        test_results.append(("TEST E", "FAIL"))
        all_passed = False

    time.sleep(2)

    # -------------------------------------------------------------
    # TEST F: Original Five Documents Verification
    # -------------------------------------------------------------
    print("\n--- TEST F: Verify Original 5 Documents Still Function ---")
    try:
        q_f = "What services does the university library provide?"
        f_res = ask_question(question=q_f, index=vector_index, llm=groq_llm)

        f_ans = f_res["answer"]
        f_sources = f_res["sources"]

        has_lib = "library_services.txt" in f_sources and len(f_ans) > 20

        print(f"Answer: {f_ans[:140]}...")
        print(f"Sources: {f_sources}")

        if has_lib:
            print("TEST F Result: PASS (Original documents continue to function perfectly)")
            test_results.append(("TEST F", "PASS"))
        else:
            print(f"TEST F Result: FAIL (Expected library_services.txt, got {f_sources})")
            test_results.append(("TEST F", "FAIL"))
            all_passed = False
    except Exception as e:
        print(f"TEST F Result: FAIL (Exception: {e})")
        test_results.append(("TEST F", "FAIL"))
        all_passed = False

    # Cleanup test uploads to restore clean baseline
    deleted = clear_uploaded_documents(index=vector_index)
    print(f"\nCleanup: Removed {deleted} test upload chunks. Knowledge base restored to baseline.")

    # Print Summary
    print("\n" + "=" * 65)
    print("KnowRAG Upload Feature Test Summary")
    print("=" * 65)
    pass_cnt = sum(1 for _, res in test_results if res == "PASS")
    for name, res in test_results:
        print(f"{name}: {res}")
    print(f"\nTotal: {pass_cnt}/{len(test_results)} PASS")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    success = run_upload_tests()
    sys.exit(0 if success else 1)
