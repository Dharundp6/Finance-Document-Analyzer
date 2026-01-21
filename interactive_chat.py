"""
Enhanced interactive chat with Financial RAG System.
Displays structured, formatted responses.
"""

import os

import requests

API_URL = "http://localhost:8000/api/v1"


# Colors for terminal (optional)
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get model information."""
    try:
        response = requests.get(f"{API_URL}/model-info")
        return response.json()
    except:
        return None


def get_stats():
    """Get system stats."""
    try:
        response = requests.get(f"{API_URL}/stats")
        return response.json()
    except:
        return None


def list_documents():
    """List all documents."""
    try:
        response = requests.get(f"{API_URL}/documents")
        return response.json()
    except:
        return []


def upload_document(file_path):
    """Upload a document."""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(f"{API_URL}/documents/upload", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ask_question(question, top_k=5):
    """Ask a question and get structured response."""
    try:
        payload = {"query": question, "top_k": top_k, "include_sources": True}
        response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be busy."}
    except Exception as e:
        return {"error": str(e)}


def format_answer(result):
    """Format and display the answer nicely."""
    if "error" in result or "detail" in result:
        error = result.get("error") or result.get("detail")
        print(f"\n{Colors.RED}❌ Error: {error}{Colors.END}")
        return

    # Header
    print(f"\n{'═' * 60}")
    print(f"{Colors.GREEN}{Colors.BOLD}📝 ANSWER{Colors.END}")
    print("═" * 60)

    # Main answer
    answer = result.get("answer", "No answer generated")
    print(f"\n{answer}")

    # Metadata
    print(f"\n{'─' * 60}")
    confidence = result.get("confidence", 0)
    time_taken = result.get("processing_time", 0)

    # Confidence bar
    conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

    print(f"{Colors.CYAN}📊 Confidence: [{conf_bar}] {confidence:.1%}{Colors.END}")
    print(f"{Colors.CYAN}⏱️  Time: {time_taken:.2f}s{Colors.END}")

    # Query analysis info (if available)
    metadata = result.get("metadata", {})
    if metadata:
        query_type = metadata.get("query_type", "unknown")
        print(f"{Colors.CYAN}🔍 Query Type: {query_type}{Colors.END}")

    # Sources
    sources = result.get("sources", [])
    if sources:
        print(f"\n{'─' * 60}")
        print(f"{Colors.YELLOW}📚 Sources ({len(sources)}):{Colors.END}")

        for i, source in enumerate(sources[:3], 1):
            score = source.get("score", 0)
            content = source.get("content", "")[:150].replace("\n", " ")
            page = source.get("page_number")

            page_info = f" (Page {page})" if page else ""
            print(f"\n   [{i}] Score: {score:.3f}{page_info}")
            print(f"       {Colors.BLUE}{content}...{Colors.END}")

    print(f"\n{'═' * 60}\n")


def show_help():
    """Show help information."""
    print(f"""
{Colors.BOLD}Available Commands:{Colors.END}

  {Colors.GREEN}[question]{Colors.END}     Ask any question about your documents

  {Colors.CYAN}upload [path]{Colors.END}  Upload a document
                   Example: upload ./report.pdf

  {Colors.CYAN}docs{Colors.END}           List all uploaded documents

  {Colors.CYAN}stats{Colors.END}          Show system statistics

  {Colors.CYAN}info{Colors.END}           Show model information

  {Colors.CYAN}clear{Colors.END}          Clear the screen

  {Colors.CYAN}help{Colors.END}           Show this help message

  {Colors.CYAN}quit{Colors.END}           Exit the program

{Colors.BOLD}Example Questions:{Colors.END}
  • What is the total revenue?
  • Compare Q1 and Q2 revenue
  • List all business segments
  • What is the profit margin?
  • Summarize the financial highlights
""")


def main():
    """Main interactive loop."""
    clear_screen()

    print(f"""
{Colors.BOLD}{"═" * 60}
   💼 FINANCIAL DOCUMENT Q&A SYSTEM
{"═" * 60}{Colors.END}
""")

    # Check server
    print("Checking server connection...")
    if not check_server():
        print(f"{Colors.RED}❌ Server is not running!{Colors.END}")
        print(f"   Start it with: {Colors.CYAN}python main.py{Colors.END}")
        return

    print(f"{Colors.GREEN}✅ Connected to server{Colors.END}")

    # Get model info
    model_info = get_model_info()
    if model_info:
        print(f"\n{Colors.CYAN}📦 Models:{Colors.END}")
        print(f"   • LLM: {model_info.get('llm_model', 'Unknown')}")
        print(f"   • Embeddings: {model_info.get('embedding_model', 'Unknown')}")
        print(f"   • Vector Store: {model_info.get('vector_store', 'Unknown')}")

    # Get stats
    stats = get_stats()
    if stats:
        print(f"\n{Colors.CYAN}📊 Status:{Colors.END}")
        print(f"   • Documents: {stats.get('total_documents', 0)}")
        print(f"   • Chunks indexed: {stats.get('total_chunks', 0)}")

    # List documents
    docs = list_documents()
    if docs:
        print(f"\n{Colors.CYAN}📁 Documents:{Colors.END}")
        for doc in docs[:5]:
            print(f"   • {doc.get('filename', 'Unknown')}")
        if len(docs) > 5:
            print(f"   ... and {len(docs) - 5} more")
    else:
        print(f"\n{Colors.YELLOW}⚠️  No documents uploaded yet{Colors.END}")
        print(f"   Use: {Colors.CYAN}upload /path/to/document.pdf{Colors.END}")

    print(f"\n{'─' * 60}")
    print(f"Type {Colors.CYAN}help{Colors.END} for commands, {Colors.CYAN}quit{Colors.END} to exit")
    print("─" * 60)

    # Main loop
    while True:
        try:
            user_input = input(f"\n{Colors.GREEN}🔍 Question:{Colors.END} ").strip()

            if not user_input:
                continue

            lower_input = user_input.lower()

            # Handle commands
            if lower_input in ["quit", "exit", "q"]:
                print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}\n")
                break

            elif lower_input == "help":
                show_help()

            elif lower_input == "clear":
                clear_screen()

            elif lower_input == "docs":
                docs = list_documents()
                if docs:
                    print(f"\n{Colors.CYAN}📁 Documents ({len(docs)}):{Colors.END}")
                    for doc in docs:
                        size = doc.get("file_size", 0)
                        size_str = (
                            f"{size / 1024:.1f} KB"
                            if size < 1024 * 1024
                            else f"{size / 1024 / 1024:.1f} MB"
                        )
                        print(f"   • {doc.get('filename')} ({size_str})")
                        print(f"     ID: {doc.get('document_id', '')[:16]}...")
                else:
                    print(f"\n{Colors.YELLOW}No documents uploaded{Colors.END}")

            elif lower_input == "stats":
                stats = get_stats()
                if stats:
                    print(f"\n{Colors.CYAN}📊 Statistics:{Colors.END}")
                    print(f"   • Documents: {stats.get('total_documents', 0)}")
                    print(f"   • Chunks: {stats.get('total_chunks', 0)}")
                    chunk_types = stats.get("chunk_types", {})
                    if chunk_types:
                        print(f"   • Chunk types: {chunk_types}")

            elif lower_input == "info":
                info = get_model_info()
                if info:
                    print(f"\n{Colors.CYAN}📦 System Information:{Colors.END}")
                    for key, value in info.items():
                        print(f"   • {key}: {value}")

            elif lower_input.startswith("upload "):
                file_path = user_input[7:].strip().strip('"').strip("'")
                print(f"\n📤 Uploading: {file_path}")
                result = upload_document(file_path)

                if result.get("success"):
                    print(f"{Colors.GREEN}✅ Upload successful!{Colors.END}")
                    print(f"   • Document ID: {result.get('document_id', '')[:16]}...")
                    print(f"   • Chunks created: {result.get('chunk_count', 0)}")
                    print(f"   • Tables found: {result.get('table_count', 0)}")
                    print(f"   • Time: {result.get('processing_time', 0):.2f}s")
                else:
                    errors = result.get("errors", [result.get("error", "Unknown error")])
                    print(f"{Colors.RED}❌ Upload failed: {errors}{Colors.END}")

            else:
                # It's a question
                print(f"\n{Colors.YELLOW}🤔 Analyzing and generating response...{Colors.END}")
                result = ask_question(user_input)
                format_answer(result)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}👋 Goodbye!{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")


if __name__ == "__main__":
    main()
