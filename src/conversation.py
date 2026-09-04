"""
KnowRAG - Conversational Routing & Intent Handling Module
---------------------------------------------------------
This module provides intent classification and routing for KnowRAG:
- Recognizes greetings (hello, hi, good morning, etc.)
- Executes slash commands (/help, /show documents, /clear, /about)
- Handles polite/casual conversation (thanks, bye, who are you?, ok)
- Routes open-ended brainstorming queries directly to conversational LLM
- Directs document-specific queries to the grounded RAG pipeline

Author/Project: KnowRAG — AI-Powered Knowledge Assistant
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from llama_index.core import VectorStoreIndex
from llama_index.llms.groq import Groq

from src.rag_pipeline import ask_question, create_rag_pipeline
from src.vector_store import get_knowledge_base_stats


# Supported slash commands
COMMAND_HELP = "/help"
COMMAND_SHOW_DOCS = "/show documents"
COMMAND_CLEAR = "/clear"
COMMAND_ABOUT = "/about"


def normalize_input(text: str) -> str:
    """Normalize input text by trimming whitespace, lowercasing, and removing trailing punctuation."""
    if not text:
        return ""
    cleaned = text.strip().lower()
    # Remove surrounding quotes and trailing punctuation for matching
    cleaned = re.sub(r"^[\"']|[\"']$", "", cleaned).strip()
    return cleaned


def detect_command(normalized_text: str) -> Optional[str]:
    """Check if the normalized input matches a supported slash command or direct command equivalent."""
    clean = normalized_text.rstrip(".!?")
    
    if clean in ["/help", "help", "/commands", "commands"]:
        return COMMAND_HELP
    elif clean in ["/show documents", "/show docs", "show documents", "/documents", "show docs", "list documents"]:
        return COMMAND_SHOW_DOCS
    elif clean in ["/clear", "clear", "/reset", "clear chat", "clear history"]:
        return COMMAND_CLEAR
    elif clean in ["/about", "about", "/info", "about knowrag"]:
        return COMMAND_ABOUT
    
    return None


def detect_greeting(normalized_text: str) -> Optional[str]:
    """Detect if the input is a standard greeting."""
    clean = normalized_text.rstrip(".!?,")
    
    # Direct exact greeting phrases
    exact_greetings = {
        "hello": "hello",
        "hi": "hi",
        "hey": "hey",
        "hii": "hii",
        "hiii": "hii",
        "heyy": "hey",
        "good morning": "good_morning",
        "good afternoon": "good_afternoon",
        "good evening": "good_evening",
        "good night": "good_night",
        "morning": "good_morning",
        "afternoon": "good_afternoon",
        "evening": "good_evening",
        "greetings": "hello",
        "howdy": "hi",
        "namaste": "hello",
    }
    
    if clean in exact_greetings:
        return exact_greetings[clean]
        
    # Pattern checks for multi-word greetings like "hello there", "hi assistant"
    if re.match(r"^(hello|hi|hey|hii)\b(\s+(there|knowrag|assistant|bot|friend|team))?$", clean):
        return "hello"
    if re.match(r"^good\s+morning\b", clean):
        return "good_morning"
    if re.match(r"^good\s+afternoon\b", clean):
        return "good_afternoon"
    if re.match(r"^good\s+evening\b", clean):
        return "good_evening"
    if re.match(r"^good\s+night\b", clean):
        return "good_night"

    return None


def detect_casual_conversation(normalized_text: str) -> Optional[str]:
    """Detect if the input is casual conversational dialogue (thanks, bye, who are you, ok)."""
    clean = normalized_text.rstrip(".!?,")
    
    # Thanks patterns
    if clean in ["thanks", "thank you", "thx", "thank u", "many thanks", "thanks a lot", "thank you so much"]:
        return "thanks"
        
    # Goodbye patterns
    if clean in ["bye", "goodbye", "see you", "see ya", "cya", "bye bye", "have a good day", "talk to you later"]:
        return "bye"
        
    # Identity / Capability queries
    identity_patterns = [
        "who are you",
        "what are you",
        "who are u",
        "what is your name",
        "what can you do",
        "tell me about yourself",
        "introduce yourself",
    ]
    if any(clean == pat or clean == f"{pat}?" for pat in identity_patterns):
        return "identity"
        
    # Acknowledgments
    if clean in ["ok", "okay", "got it", "cool", "great", "awesome", "understood", "alright", "all right", "perfect", "sounds good"]:
        return "acknowledgment"
        
    return None


def is_brainstorming_query(normalized_text: str) -> bool:
    """Detect if the user is asking for open-ended brainstorming, ideation, or general creative suggestions."""
    clean = normalized_text.rstrip(".!?")
    
    # Explicit brainstorming trigger keywords
    brainstorm_keywords = [
        "brainstorm",
        "project idea",
        "project ideas",
        "give me ideas",
        "suggest ideas",
        "help me with ideas",
        "what can i build",
        "what should i build",
        "give me some ideas",
        "give me cybersecurity project ideas",
        "give me ai project ideas",
        "give me python project ideas",
        "ideas for",
        "suggest some topics",
        "generate ideas",
    ]
    
    for kw in brainstorm_keywords:
        if kw in clean:
            return True
            
    # Generic question patterns not directed at knowledge base documents
    if re.search(r"\b(give|suggest|provide|recommend)\s+(me\s+)?(some\s+)?(project\s+)?ideas\b", clean):
        return True
    if re.search(r"\bwhat\s+(projects?|apps?|tools?)\s+can\s+i\s+(build|make|create|develop)\b", clean):
        return True

    return False


def get_greeting_response(greeting_type: str) -> str:
    """Return friendly, short, natural responses for greetings."""
    if greeting_type == "hi":
        return "Hello! 😊 I'm KnowRAG. How can I help you today?"
    elif greeting_type == "good_morning":
        return "Good morning! ☀️ How can I help you today?"
    elif greeting_type == "good_afternoon":
        return "Good afternoon! ☀️ How can I help you today?"
    elif greeting_type == "good_evening":
        return "Good evening! 🌙 How can I help you today?"
    elif greeting_type == "good_night":
        return "Good night! 🌙 Let me know if you need anything before you go."
    else:  # default hello / hey / hii
        return "Hi! 👋 How can I help you today? You can ask me questions about the knowledge base, upload a document, or brainstorm ideas with me."


def get_casual_response(casual_type: str) -> str:
    """Return friendly responses for casual dialogue."""
    if casual_type == "thanks":
        return "You're welcome! 😊 Let me know if you need anything else."
    elif casual_type == "bye":
        return "Goodbye! 👋 Have a great day!"
    elif casual_type == "identity":
        return "I'm KnowRAG, an AI-powered knowledge assistant. 🤖 I can answer questions using information from your uploaded documents and the university knowledge base."
    elif casual_type == "acknowledgment":
        return "Got it! Let me know what you'd like to explore next. 😊"
    return "How else can I assist you today?"


def handle_command(command: str) -> Dict[str, Any]:
    """Execute built-in commands and return structured responses."""
    if command == COMMAND_HELP:
        help_text = (
            "Here are some things you can do:\n\n"
            "📚 **Ask questions** about your uploaded documents\n"
            "📄 **Upload** PDF or TXT documents\n"
            "🔎 **Get answers** grounded in your documents\n"
            "📌 **View sources** used for an answer\n"
            "🧠 **Brainstorm ideas** and ask general questions\n\n"
            "**Commands:**\n"
            "• `/help` — Show available commands\n"
            "• `/show documents` — Show uploaded documents\n"
            "• `/clear` — Clear chat history\n"
            "• `/about` — Show information about KnowRAG"
        )
        return {
            "answer": help_text,
            "sources": [],
            "source_details": [],
            "intent": "command",
            "command": COMMAND_HELP,
        }

    elif command == COMMAND_SHOW_DOCS:
        try:
            stats = get_knowledge_base_stats()
            default_docs = [
                "academic_programs.txt",
                "campus_rules.txt",
                "library_services.txt",
                "student_services.txt",
                "university_overview.txt",
            ]
            uploaded = stats.get("uploaded_documents", [])
            
            lines = ["📚 **Available Documents:**\n"]
            for doc in default_docs:
                lines.append(f"• `{doc}`")
            
            if uploaded:
                lines.append("\n**User Uploaded Documents:**")
                for u_doc in uploaded:
                    fn = u_doc.get("filename", "unknown")
                    chunks = u_doc.get("chunks", 0)
                    lines.append(f"• `📄 {fn}` ({chunks} chunks)")

            lines.append("\nYou can also upload additional PDF or TXT documents.")
            docs_text = "\n".join(lines)
        except Exception as e:
            docs_text = (
                "📚 **Available Documents:**\n\n"
                "• `academic_programs.txt`\n"
                "• `campus_rules.txt`\n"
                "• `library_services.txt`\n"
                "• `student_services.txt`\n"
                "• `university_overview.txt`\n\n"
                "You can also upload additional PDF or TXT documents."
            )

        return {
            "answer": docs_text,
            "sources": [],
            "source_details": [],
            "intent": "command",
            "command": COMMAND_SHOW_DOCS,
        }

    elif command == COMMAND_CLEAR:
        return {
            "answer": "🧹 Chat history cleared. How can I help you?",
            "sources": [],
            "source_details": [],
            "intent": "command",
            "command": COMMAND_CLEAR,
            "action": "clear_chat",
        }

    elif command == COMMAND_ABOUT:
        about_text = (
            "🤖 **KnowRAG — AI-Powered Knowledge Assistant**\n\n"
            "KnowRAG uses Retrieval-Augmented Generation (RAG) to answer questions using information "
            "retrieved from the available knowledge base.\n\n"
            "**Pipeline:**\n"
            "Documents → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Grounded Answer\n\n"
            "The system is designed to reduce hallucinations by grounding answers in the available documents."
        )
        return {
            "answer": about_text,
            "sources": [],
            "source_details": [],
            "intent": "command",
            "command": COMMAND_ABOUT,
        }

    return {
        "answer": f"Unknown command: `{command}`. Type `/help` for available commands.",
        "sources": [],
        "source_details": [],
        "intent": "command",
    }


def generate_brainstorming_response(prompt: str, llm: Groq) -> str:
    """Generate a creative, helpful conversational response using the LLM without RAG context."""
    system_prompt = (
        "You are KnowRAG, a helpful and knowledgeable AI Assistant.\n"
        "The user is asking for brainstorming assistance, project ideas, or general creative guidance.\n"
        "Provide a clear, well-structured, inspiring, and actionable response.\n"
        "Do not claim that these ideas come from a specific document."
    )
    formatted_prompt = f"{system_prompt}\n\nUSER PROMPT: {prompt}\n\nRESPONSE:"
    response = llm.complete(formatted_prompt)
    return response.text.strip()


def process_user_query(
    query: str,
    index: Optional[VectorStoreIndex] = None,
    llm: Optional[Groq] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Main conversational router that detects commands, greetings, casual dialogue,
    brainstorming queries, and knowledge-base RAG queries.

    Flow:
    User Input -> Normalize -> Command Check -> Greeting Check -> Casual Check -> Brainstorm Check -> RAG Pipeline

    Args:
        query (str): The raw input message from the user.
        index (Optional[VectorStoreIndex]): The active LlamaIndex vector store.
        llm (Optional[Groq]): The active Groq LLM instance.
        top_k (int): Top-k retrieval count for RAG queries.

    Returns:
        Dict[str, Any]: Structured response containing 'answer', 'sources', 'source_details', and 'intent'.
    """
    if not query or not query.strip():
        return {
            "question": query,
            "answer": "Please enter a question, command (e.g. `/help`), or greeting!",
            "sources": [],
            "source_details": [],
            "intent": "empty",
        }

    normalized = normalize_input(query)

    # 1. Check Slash Commands
    command = detect_command(normalized)
    if command:
        res = handle_command(command)
        res["question"] = query
        return res

    # 2. Check Greetings
    greeting = detect_greeting(normalized)
    if greeting:
        return {
            "question": query,
            "answer": get_greeting_response(greeting),
            "sources": [],
            "source_details": [],
            "intent": "greeting",
        }

    # 3. Check Casual Dialogue
    casual = detect_casual_conversation(normalized)
    if casual:
        return {
            "question": query,
            "answer": get_casual_response(casual),
            "sources": [],
            "source_details": [],
            "intent": "casual",
        }

    # 4. Check Brainstorming / Creative Inquiries
    if is_brainstorming_query(normalized):
        active_llm = llm if llm is not None else create_rag_pipeline(temperature=0.7)[1]
        brainstorm_ans = generate_brainstorming_response(prompt=query, llm=active_llm)
        return {
            "question": query,
            "answer": brainstorm_ans,
            "sources": [],
            "source_details": [],
            "intent": "brainstorming",
        }

    # 5. Fallback: Core Grounded RAG Pipeline for Knowledge Base / Document Query
    rag_result = ask_question(question=query, top_k=top_k, index=index, llm=llm)
    rag_result["intent"] = "rag"
    return rag_result


if __name__ == "__main__":
    print("=" * 60)
    print("KnowRAG Conversational Intent Routing Test")
    print("=" * 60)
    
    test_inputs = [
        "hello",
        "Hi",
        "good morning",
        "/help",
        "/show documents",
        "/about",
        "thanks",
        "bye",
        "who are you?",
        "give me 3 cybersecurity project ideas",
        "What services does the university library provide?",
    ]

    vector_index, groq_llm = create_rag_pipeline(temperature=0.0)

    for inp in test_inputs:
        print(f"\nUser Input: \"{inp}\"")
        res = process_user_query(inp, index=vector_index, llm=groq_llm)
        print(f"Detected Intent : {res.get('intent')}")
        print(f"Sources Count   : {len(res.get('sources', []))}")
        print(f"Answer Preview  : {res.get('answer', '')[:120]}...")
