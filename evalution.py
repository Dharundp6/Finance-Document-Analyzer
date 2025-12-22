"""
Run RAG system evaluation.
"""

import argparse
import requests
import json
from pathlib import Path

from evaluation.evaluator import RAGEvaluator
from evaluation.dataset import EvaluationDataset, create_financial_evaluation_dataset
from evaluation.metrics import RAGMetrics
from config.logging_config import logger


API_URL = "http://localhost:8000/api/v1"


def query_rag_system(query: str) -> dict:
    """Query the RAG system API."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query, "top_k": 5, "include_sources": True},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": []}


def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_evaluation(
    dataset_path: str = None,
    output_path: str = None,
    create_sample: bool = False
):
    """Run the evaluation."""
    
    print("\n" + "=" * 70)
    print("   🔬 RAG SYSTEM EVALUATION")
    print("=" * 70)
    
    # Check server
    print("\n1️⃣  Checking server connection...")
    if not check_server():
        print("❌ Server is not running! Start it with: python main.py")
        return
    print("✅ Server is connected")
    
    # Load or create dataset
    print("\n2️⃣  Loading evaluation dataset...")
    
    if create_sample:
        # Create and save sample dataset
        dataset = create_financial_evaluation_dataset()
        sample_path = "evaluation/sample_dataset.json"
        Path("evaluation").mkdir(exist_ok=True)
        dataset.save(sample_path)
        print(f"✅ Sample dataset created and saved to {sample_path}")
        print(f"   Edit this file to customize test cases for your documents.")
        return
    
    if dataset_path and Path(dataset_path).exists():
        dataset = EvaluationDataset.load(dataset_path)
    else:
        print("   Using default financial evaluation dataset")
        dataset = create_financial_evaluation_dataset()
    
    print(f"✅ Loaded {len(dataset)} test cases")
    
    # Run evaluation
    print("\n3️⃣  Running evaluation...")
    
    evaluator = RAGEvaluator(query_function=query_rag_system)
    report = evaluator.evaluate_dataset(dataset, verbose=True)
    
    # Print summary
    report.print_summary()
    
    # Save report
    if output_path:
        report.save(output_path)
        print(f"\n📄 Full report saved to: {output_path}")
    else:
        # Default output path
        output_path = f"evaluation/report_{report.timestamp[:10]}.json"
        Path("evaluation").mkdir(exist_ok=True)
        report.save(output_path)
        print(f"\n📄 Full report saved to: {output_path}")
    
    print("\n✅ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to evaluation dataset JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save evaluation report"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample evaluation dataset"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        create_sample=args.create_sample
    )


if __name__ == "__main__":
    main()