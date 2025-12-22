"""
RAG System Evaluator.
Runs evaluation tests and generates comprehensive reports.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from config.logging_config import logger
from evaluation.metrics import (
    RAGMetrics,
    RetrievalMetrics,
    GenerationMetrics,
    EndToEndMetrics,
    EvaluationResult
)
from evaluation.dataset import EvaluationDataset, TestCase


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    dataset_name: str
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Aggregate metrics
    avg_retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    avg_generation_metrics: Dict[str, float] = field(default_factory=dict)
    avg_end_to_end_metrics: Dict[str, float] = field(default_factory=dict)
    
    # By query type
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # By difficulty
    metrics_by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Individual results
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance stats
    total_time_seconds: float = 0.0
    avg_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "pass_rate": f"{self.passed_tests / max(self.total_tests, 1) * 100:.1f}%"
            },
            "average_metrics": {
                "retrieval": self.avg_retrieval_metrics,
                "generation": self.avg_generation_metrics,
                "end_to_end": self.avg_end_to_end_metrics
            },
            "metrics_by_query_type": self.metrics_by_type,
            "metrics_by_difficulty": self.metrics_by_difficulty,
            "performance": {
                "total_time_seconds": round(self.total_time_seconds, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2)
            },
            "detailed_results": self.results
        }
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")
    
    def print_summary(self) -> None:
        """Print a summary of the evaluation."""
        print("\n" + "=" * 70)
        print(f"📊 EVALUATION REPORT: {self.dataset_name}")
        print("=" * 70)
        
        print(f"\n📋 Summary:")
        print(f"   • Total Tests: {self.total_tests}")
        print(f"   • Passed: {self.passed_tests} ({self.passed_tests/max(self.total_tests,1)*100:.1f}%)")
        print(f"   • Failed: {self.failed_tests}")
        print(f"   • Total Time: {self.total_time_seconds:.2f}s")
        print(f"   • Avg Latency: {self.avg_latency_ms:.0f}ms")
        
        print(f"\n📈 Retrieval Metrics:")
        for metric, value in self.avg_retrieval_metrics.items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"   • {metric:15}: [{bar}] {value:.3f}")
        
        print(f"\n📝 Generation Metrics:")
        for metric, value in self.avg_generation_metrics.items():
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"   • {metric:15}: [{bar}] {value:.3f}")
        
        print(f"\n🎯 End-to-End Metrics:")
        for metric, value in self.avg_end_to_end_metrics.items():
            if isinstance(value, float) and value <= 1.0:
                bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
                print(f"   • {metric:18}: [{bar}] {value:.3f}")
            else:
                print(f"   • {metric:18}: {value}")
        
        print("\n" + "=" * 70)


class RAGEvaluator:
    """
    Evaluates RAG system performance.
    """
    
    def __init__(
        self,
        query_function,
        retrieval_function=None
    ):
        """
        Initialize the evaluator.
        
        Args:
            query_function: Function that takes a query and returns (answer, sources)
            retrieval_function: Optional function for direct retrieval testing
        """
        self.query_fn = query_function
        self.retrieval_fn = retrieval_function
        self.metrics = RAGMetrics()
        
        logger.info("RAGEvaluator initialized")
    
    def evaluate_single(
        self,
        test_case: TestCase,
        timeout: float = 120.0
    ) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: The test case to evaluate
            timeout: Maximum time for the query
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating: {test_case.query[:50]}...")
        
        start_time = time.time()
        
        try:
            # Execute query
            result = self.query_fn(test_case.query)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract data from result
            generated_answer = result.get('answer', '')
            sources = result.get('sources', [])
            
            # Get retrieved chunk contents
            retrieved_chunks = [s.get('content', '') for s in sources]
            retrieved_ids = [s.get('chunk_id', '') for s in sources]
            
            # Context for faithfulness
            context = '\n'.join(retrieved_chunks)
            
            # Calculate retrieval metrics
            retrieval_metrics = self.metrics.calculate_retrieval_metrics(
                retrieved=retrieved_ids,
                relevant=test_case.relevant_chunk_ids or [],
                total_documents=100
            )
            
            # If no relevant chunks defined, estimate from keywords
            if not test_case.relevant_chunk_ids and test_case.relevant_keywords:
                retrieval_metrics = self._estimate_retrieval_from_keywords(
                    retrieved_chunks,
                    test_case.relevant_keywords
                )
            
            # Calculate generation metrics
            generation_metrics = self.metrics.calculate_generation_metrics(
                generated=generated_answer,
                reference=test_case.expected_answer,
                query=test_case.query,
                context=context
            )
            
            # Calculate end-to-end metrics
            end_to_end_metrics = self.metrics.calculate_end_to_end_metrics(
                query=test_case.query,
                generated_answer=generated_answer,
                expected_answer=test_case.expected_answer,
                retrieved_chunks=retrieved_chunks,
                relevant_chunks=test_case.relevant_chunk_ids,
                latency_ms=latency_ms,
                tokens_used=result.get('token_usage', {}).get('completion_tokens', 0)
            )
            
            return EvaluationResult(
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                generated_answer=generated_answer,
                retrieved_chunks=retrieved_chunks,
                relevant_chunks=test_case.relevant_chunk_ids,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
                end_to_end_metrics=end_to_end_metrics,
                metadata={
                    "query_type": test_case.query_type,
                    "difficulty": test_case.difficulty,
                    "test_id": test_case.id
                }
            )
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            
            # Return failed result
            return EvaluationResult(
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                generated_answer=f"ERROR: {str(e)}",
                retrieved_chunks=[],
                relevant_chunks=test_case.relevant_chunk_ids,
                retrieval_metrics=RetrievalMetrics(),
                generation_metrics=GenerationMetrics(),
                end_to_end_metrics=EndToEndMetrics(
                    latency_ms=(time.time() - start_time) * 1000
                ),
                metadata={
                    "query_type": test_case.query_type,
                    "difficulty": test_case.difficulty,
                    "test_id": test_case.id,
                    "error": str(e)
                }
            )
    
    def _estimate_retrieval_from_keywords(
        self,
        retrieved_chunks: List[str],
        keywords: List[str]
    ) -> RetrievalMetrics:
        """Estimate retrieval metrics based on keyword presence."""
        if not retrieved_chunks or not keywords:
            return RetrievalMetrics()
        
        keywords_lower = [k.lower() for k in keywords]
        
        relevant_chunks = []
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            matching_keywords = sum(1 for kw in keywords_lower if kw in chunk_lower)
            if matching_keywords >= len(keywords) * 0.3:  # 30% threshold
                relevant_chunks.append(chunk)
        
        precision = len(relevant_chunks) / len(retrieved_chunks)
        recall = min(len(relevant_chunks) / max(len(keywords), 1), 1.0)
        
        f1 = self.metrics.calculate_f1(precision, recall)
        
        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            hit_rate=1.0 if relevant_chunks else 0.0
        )
    
    def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        verbose: bool = True
    ) -> EvaluationReport:
        """
        Evaluate an entire dataset.
        
        Args:
            dataset: The evaluation dataset
            verbose: Whether to print progress
            
        Returns:
            Complete evaluation report
        """
        logger.info(f"Starting evaluation of {len(dataset)} test cases")
        
        start_time = time.time()
        results: List[EvaluationResult] = []
        
        for i, test_case in enumerate(dataset):
            if verbose:
                print(f"\r  Evaluating {i+1}/{len(dataset)}: {test_case.query[:40]}...", end="")
            
            result = self.evaluate_single(test_case)
            results.append(result)
        
        if verbose:
            print()  # New line after progress
        
        total_time = time.time() - start_time
        
        # Generate report
        report = self._generate_report(dataset.name, results, total_time)
        
        logger.info(f"Evaluation completed in {total_time:.2f}s")
        
        return report
    
    def _generate_report(
        self,
        dataset_name: str,
        results: List[EvaluationResult],
        total_time: float
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        
        # Calculate pass/fail
        passed = sum(1 for r in results if r.end_to_end_metrics.answer_correctness > 0.5)
        failed = len(results) - passed
        
        # Aggregate retrieval metrics
        retrieval_metrics = {
            "precision": statistics.mean([r.retrieval_metrics.precision for r in results]),
            "recall": statistics.mean([r.retrieval_metrics.recall for r in results]),
            "f1_score": statistics.mean([r.retrieval_metrics.f1_score for r in results]),
            "mrr": statistics.mean([r.retrieval_metrics.mrr for r in results]),
            "map": statistics.mean([r.retrieval_metrics.map_score for r in results]),
            "ndcg": statistics.mean([r.retrieval_metrics.ndcg for r in results]),
            "hit_rate": statistics.mean([r.retrieval_metrics.hit_rate for r in results])
        }
        
        # Aggregate generation metrics
        generation_metrics = {
            "bleu": statistics.mean([r.generation_metrics.bleu_score for r in results]),
            "rouge_1": statistics.mean([r.generation_metrics.rouge_1 for r in results]),
            "rouge_2": statistics.mean([r.generation_metrics.rouge_2 for r in results]),
            "rouge_l": statistics.mean([r.generation_metrics.rouge_l for r in results]),
            "answer_relevance": statistics.mean([r.generation_metrics.answer_relevance for r in results]),
            "faithfulness": statistics.mean([r.generation_metrics.faithfulness for r in results]),
            "completeness": statistics.mean([r.generation_metrics.answer_completeness for r in results])
        }
        
        # Aggregate end-to-end metrics
        e2e_metrics = {
            "answer_correctness": statistics.mean([r.end_to_end_metrics.answer_correctness for r in results]),
            "context_precision": statistics.mean([r.end_to_end_metrics.context_precision for r in results]),
            "context_recall": statistics.mean([r.end_to_end_metrics.context_recall for r in results]),
            "avg_latency_ms": statistics.mean([r.end_to_end_metrics.latency_ms for r in results])
        }
        
        # Metrics by query type
        metrics_by_type = {}
        query_types = set(r.metadata.get('query_type', 'unknown') for r in results)
        
        for qtype in query_types:
            type_results = [r for r in results if r.metadata.get('query_type') == qtype]
            if type_results:
                metrics_by_type[qtype] = {
                    "count": len(type_results),
                    "avg_correctness": statistics.mean([r.end_to_end_metrics.answer_correctness for r in type_results]),
                    "avg_relevance": statistics.mean([r.generation_metrics.answer_relevance for r in type_results])
                }
        
        # Metrics by difficulty
        metrics_by_difficulty = {}
        difficulties = set(r.metadata.get('difficulty', 'unknown') for r in results)
        
        for diff in difficulties:
            diff_results = [r for r in results if r.metadata.get('difficulty') == diff]
            if diff_results:
                metrics_by_difficulty[diff] = {
                    "count": len(diff_results),
                    "avg_correctness": statistics.mean([r.end_to_end_metrics.answer_correctness for r in diff_results]),
                    "pass_rate": sum(1 for r in diff_results if r.end_to_end_metrics.answer_correctness > 0.5) / len(diff_results)
                }
        
        return EvaluationReport(
            dataset_name=dataset_name,
            timestamp=datetime.utcnow().isoformat(),
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            avg_retrieval_metrics=retrieval_metrics,
            avg_generation_metrics=generation_metrics,
            avg_end_to_end_metrics=e2e_metrics,
            metrics_by_type=metrics_by_type,
            metrics_by_difficulty=metrics_by_difficulty,
            results=[r.to_dict() for r in results],
            total_time_seconds=total_time,
            avg_latency_ms=e2e_metrics["avg_latency_ms"]
        )