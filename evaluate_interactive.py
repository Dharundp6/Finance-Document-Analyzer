"""
Interactive evaluation tool for testing individual queries.
"""

import requests

from evaluation.metrics import RAGMetrics

API_URL = "http://localhost:8000/api/v1"


def evaluate_single_query():
    """Interactively evaluate a single query."""

    metrics = RAGMetrics()

    print("\n" + "=" * 60)
    print("   🔬 SINGLE QUERY EVALUATION")
    print("=" * 60)

    # Get query
    query = input("\n📝 Enter your query: ").strip()
    if not query:
        print("No query provided")
        return

    # Get expected answer
    expected = input("📝 Enter expected answer: ").strip()
    if not expected:
        print("No expected answer provided")
        return

    # Query the system
    print("\n🔄 Querying RAG system...")

    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query, "top_k": 5, "include_sources": True},
            timeout=120,
        )
        result = response.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    generated = result.get("answer", "")
    sources = result.get("sources", [])
    context = "\n".join([s.get("content", "") for s in sources])

    # Calculate metrics
    print("\n📊 Calculating metrics...")

    # Generation metrics
    rouge_1, rouge_2, rouge_l = metrics.calculate_rouge(generated, expected)
    bleu = metrics.calculate_bleu(generated, expected)
    relevance = metrics.calculate_answer_relevance(query, generated)
    faithfulness = metrics.calculate_faithfulness(generated, context)
    completeness = metrics.calculate_answer_completeness(generated, expected)
    correctness = metrics.calculate_answer_correctness(generated, expected)

    # Display results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n🔹 Query: {query}")
    print(f"\n🔹 Expected Answer:\n   {expected[:200]}...")
    print(f"\n🔹 Generated Answer:\n   {generated[:200]}...")

    print(f"\n{'─' * 60}")
    print("📈 METRICS:")
    print(f"{'─' * 60}")

    metrics_display = [
        ("Answer Correctness", correctness),
        ("Answer Relevance", relevance),
        ("Faithfulness", faithfulness),
        ("Completeness", completeness),
        ("BLEU Score", bleu),
        ("ROUGE-1", rouge_1),
        ("ROUGE-2", rouge_2),
        ("ROUGE-L", rouge_l),
    ]

    for name, value in metrics_display:
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        status = "✅" if value >= 0.5 else "⚠️" if value >= 0.3 else "❌"
        print(f"   {status} {name:20}: [{bar}] {value:.3f}")

    # Overall assessment
    avg_score = (correctness + relevance + faithfulness + completeness) / 4

    print(f"\n{'─' * 60}")
    print(f"📊 OVERALL SCORE: {avg_score:.1%}")

    if avg_score >= 0.7:
        print("   ✅ EXCELLENT - The answer is accurate and complete")
    elif avg_score >= 0.5:
        print("   ⚠️  GOOD - The answer is mostly correct but could be improved")
    elif avg_score >= 0.3:
        print("   ⚠️  FAIR - The answer needs improvement")
    else:
        print("   ❌ POOR - The answer is incorrect or irrelevant")

    print("=" * 60)


def main():
    while True:
        evaluate_single_query()

        again = input("\n\nEvaluate another query? (y/n): ").strip().lower()
        if again != "y":
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
