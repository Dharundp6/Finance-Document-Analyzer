"""
Evaluation metrics for RAG system.
Includes retrieval metrics, generation metrics, and end-to-end metrics.
"""

import re
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from config.logging_config import logger


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg": round(self.ndcg, 4),
            "hit_rate": round(self.hit_rate, 4)
        }


@dataclass
class GenerationMetrics:
    """Metrics for generation quality."""
    bleu_score: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_completeness: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "bleu": round(self.bleu_score, 4),
            "rouge_1": round(self.rouge_1, 4),
            "rouge_2": round(self.rouge_2, 4),
            "rouge_l": round(self.rouge_l, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "faithfulness": round(self.faithfulness, 4),
            "completeness": round(self.answer_completeness, 4)
        }


@dataclass
class EndToEndMetrics:
    """End-to-end RAG system metrics."""
    answer_correctness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_similarity: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_correctness": round(self.answer_correctness, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_similarity": round(self.answer_similarity, 4),
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""
    query: str
    expected_answer: str
    generated_answer: str
    retrieved_chunks: List[str]
    relevant_chunks: List[str]
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    end_to_end_metrics: EndToEndMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "expected_answer": self.expected_answer[:200] + "..." if len(self.expected_answer) > 200 else self.expected_answer,
            "generated_answer": self.generated_answer[:200] + "..." if len(self.generated_answer) > 200 else self.generated_answer,
            "retrieval": self.retrieval_metrics.to_dict(),
            "generation": self.generation_metrics.to_dict(),
            "end_to_end": self.end_to_end_metrics.to_dict(),
            "metadata": self.metadata
        }


class RAGMetrics:
    """
    Calculate various metrics for RAG system evaluation.
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Avoid division by zero
    
    # =========================================
    # RETRIEVAL METRICS
    # =========================================
    
    def calculate_precision(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Precision = Relevant Retrieved / Total Retrieved
        
        How many retrieved documents are actually relevant?
        """
        if not retrieved:
            return 0.0
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set)
        
        return precision
    
    def calculate_recall(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Recall = Relevant Retrieved / Total Relevant
        
        How many relevant documents were retrieved?
        """
        if not relevant:
            return 0.0
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set & relevant_set)
        recall = true_positives / len(relevant_set)
        
        return recall
    
    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        
        Harmonic mean of precision and recall.
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_accuracy(
        self,
        retrieved: List[str],
        relevant: List[str],
        total_documents: int
    ) -> float:
        """
        Accuracy = (TP + TN) / Total
        
        Overall correctness of retrieval.
        """
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        true_positives = len(retrieved_set & relevant_set)
        true_negatives = total_documents - len(retrieved_set | relevant_set)
        
        accuracy = (true_positives + true_negatives) / max(total_documents, 1)
        
        return accuracy
    
    def calculate_mrr(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Mean Reciprocal Rank (MRR)
        
        MRR = 1 / rank of first relevant document
        """
        relevant_set = set(relevant)
        
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_map(
        self,
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Mean Average Precision (MAP)
        
        Average of precision at each relevant document.
        """
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        precisions = []
        relevant_count = 0
        
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precisions.append(precision_at_k)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_set)
    
    def calculate_ndcg(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG)
        
        Measures ranking quality with position discount.
        """
        if not retrieved or not relevant:
            return 0.0
        
        # Default binary relevance if scores not provided
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant}
        
        # Calculate DCG
        dcg = 0.0
        for rank, doc in enumerate(retrieved, 1):
            rel = relevance_scores.get(doc, 0.0)
            dcg += rel / math.log2(rank + 1)
        
        # Calculate ideal DCG
        ideal_scores = sorted(
            [relevance_scores.get(doc, 0.0) for doc in relevant],
            reverse=True
        )
        
        idcg = 0.0
        for rank, rel in enumerate(ideal_scores, 1):
            idcg += rel / math.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_hit_rate(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int = None
    ) -> float:
        """
        Hit Rate @ K
        
        Did we retrieve at least one relevant document in top K?
        """
        if k is None:
            k = len(retrieved)
        
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        return 1.0 if retrieved_k & relevant_set else 0.0
    
    def calculate_retrieval_metrics(
        self,
        retrieved: List[str],
        relevant: List[str],
        total_documents: int = 100
    ) -> RetrievalMetrics:
        """Calculate all retrieval metrics."""
        precision = self.calculate_precision(retrieved, relevant)
        recall = self.calculate_recall(retrieved, relevant)
        
        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=self.calculate_f1(precision, recall),
            accuracy=self.calculate_accuracy(retrieved, relevant, total_documents),
            mrr=self.calculate_mrr(retrieved, relevant),
            map_score=self.calculate_map(retrieved, relevant),
            ndcg=self.calculate_ndcg(retrieved, relevant),
            hit_rate=self.calculate_hit_rate(retrieved, relevant)
        )
    
    # =========================================
    # GENERATION METRICS
    # =========================================
    
    def calculate_bleu(
        self,
        generated: str,
        reference: str,
        max_n: int = 4
    ) -> float:
        """
        BLEU Score (Bilingual Evaluation Understudy)
        
        Measures n-gram overlap between generated and reference text.
        """
        generated_tokens = self._tokenize(generated)
        reference_tokens = self._tokenize(reference)
        
        if not generated_tokens or not reference_tokens:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        
        for n in range(1, max_n + 1):
            gen_ngrams = self._get_ngrams(generated_tokens, n)
            ref_ngrams = self._get_ngrams(reference_tokens, n)
            
            if not gen_ngrams:
                precisions.append(0.0)
                continue
            
            gen_counts = Counter(gen_ngrams)
            ref_counts = Counter(ref_ngrams)
            
            matches = sum(min(gen_counts[ng], ref_counts[ng]) for ng in gen_counts)
            precision = matches / len(gen_ngrams)
            precisions.append(precision)
        
        # Geometric mean of precisions
        if 0 in precisions:
            return 0.0
        
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        
        # Brevity penalty
        bp = 1.0
        if len(generated_tokens) < len(reference_tokens):
            bp = math.exp(1 - len(reference_tokens) / len(generated_tokens))
        
        return bp * math.exp(log_precision)
    
    def calculate_rouge(
        self,
        generated: str,
        reference: str
    ) -> Tuple[float, float, float]:
        """
        ROUGE Scores (Recall-Oriented Understudy for Gisting Evaluation)
        
        Returns: (ROUGE-1, ROUGE-2, ROUGE-L)
        """
        generated_tokens = self._tokenize(generated)
        reference_tokens = self._tokenize(reference)
        
        if not generated_tokens or not reference_tokens:
            return 0.0, 0.0, 0.0
        
        # ROUGE-1 (unigram)
        rouge_1 = self._rouge_n(generated_tokens, reference_tokens, 1)
        
        # ROUGE-2 (bigram)
        rouge_2 = self._rouge_n(generated_tokens, reference_tokens, 2)
        
        # ROUGE-L (longest common subsequence)
        rouge_l = self._rouge_l(generated_tokens, reference_tokens)
        
        return rouge_1, rouge_2, rouge_l
    
    def _rouge_n(
        self,
        generated: List[str],
        reference: List[str],
        n: int
    ) -> float:
        """Calculate ROUGE-N F1 score."""
        gen_ngrams = Counter(self._get_ngrams(generated, n))
        ref_ngrams = Counter(self._get_ngrams(reference, n))
        
        if not gen_ngrams or not ref_ngrams:
            return 0.0
        
        overlap = sum(min(gen_ngrams[ng], ref_ngrams[ng]) for ng in gen_ngrams)
        
        precision = overlap / sum(gen_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _rouge_l(
        self,
        generated: List[str],
        reference: List[str]
    ) -> float:
        """Calculate ROUGE-L F1 score using LCS."""
        lcs_length = self._lcs_length(generated, reference)
        
        if not generated or not reference:
            return 0.0
        
        precision = lcs_length / len(generated)
        recall = lcs_length / len(reference)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """Calculate length of Longest Common Subsequence."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Answer Relevance Score
        
        Measures how relevant the answer is to the query.
        Based on keyword overlap and semantic similarity proxy.
        """
        query_tokens = set(self._tokenize(query.lower()))
        answer_tokens = set(self._tokenize(answer.lower()))
        
        if not query_tokens or not answer_tokens:
            return 0.0
        
        # Remove stopwords for better relevance measure
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'under', 'again', 'further', 'then', 'once',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                    'those', 'am', 'or', 'and', 'but', 'if', 'because', 'until',
                    'while', 'although', 'though', 'after', 'before'}
        
        query_keywords = query_tokens - stopwords
        answer_keywords = answer_tokens - stopwords
        
        if not query_keywords:
            return 0.5  # Neutral if no keywords
        
        overlap = len(query_keywords & answer_keywords)
        relevance = overlap / len(query_keywords)
        
        return min(relevance, 1.0)
    
    def calculate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Faithfulness Score
        
        Measures how much of the answer is grounded in the context.
        Higher = answer is more faithful to source documents.
        """
        answer_tokens = set(self._tokenize(answer.lower()))
        context_tokens = set(self._tokenize(context.lower()))
        
        if not answer_tokens:
            return 0.0
        
        if not context_tokens:
            return 0.0
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                       'been', 'have', 'has', 'had', 'do', 'does', 'did',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by'}
        
        answer_content = answer_tokens - common_words
        context_content = context_tokens - common_words
        
        if not answer_content:
            return 1.0  # Empty answer is vacuously faithful
        
        grounded = len(answer_content & context_content)
        faithfulness = grounded / len(answer_content)
        
        return faithfulness
    
    def calculate_answer_completeness(
        self,
        generated: str,
        expected: str
    ) -> float:
        """
        Answer Completeness Score
        
        Measures how complete the generated answer is compared to expected.
        """
        expected_tokens = set(self._tokenize(expected.lower()))
        generated_tokens = set(self._tokenize(generated.lower()))
        
        if not expected_tokens:
            return 1.0
        
        # Key terms from expected answer
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
                    'in', 'for', 'on', 'with', 'at', 'by', 'and', 'or', 'but'}
        
        expected_keywords = expected_tokens - stopwords
        generated_keywords = generated_tokens - stopwords
        
        if not expected_keywords:
            return 1.0
        
        covered = len(expected_keywords & generated_keywords)
        completeness = covered / len(expected_keywords)
        
        return completeness
    
    def calculate_generation_metrics(
        self,
        generated: str,
        reference: str,
        query: str,
        context: str
    ) -> GenerationMetrics:
        """Calculate all generation metrics."""
        rouge_1, rouge_2, rouge_l = self.calculate_rouge(generated, reference)
        
        return GenerationMetrics(
            bleu_score=self.calculate_bleu(generated, reference),
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            answer_relevance=self.calculate_answer_relevance(query, generated),
            faithfulness=self.calculate_faithfulness(generated, context),
            answer_completeness=self.calculate_answer_completeness(generated, reference)
        )
    
    # =========================================
    # END-TO-END METRICS
    # =========================================
    
    def calculate_answer_correctness(
        self,
        generated: str,
        expected: str
    ) -> float:
        """
        Answer Correctness Score
        
        Combines semantic similarity and factual accuracy.
        """
        # Tokenize
        gen_tokens = self._tokenize(generated.lower())
        exp_tokens = self._tokenize(expected.lower())
        
        if not gen_tokens or not exp_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        gen_set = set(gen_tokens)
        exp_set = set(exp_tokens)
        
        jaccard = len(gen_set & exp_set) / len(gen_set | exp_set)
        
        # Calculate cosine similarity using term frequency
        all_words = gen_set | exp_set
        
        gen_vec = [gen_tokens.count(w) for w in all_words]
        exp_vec = [exp_tokens.count(w) for w in all_words]
        
        dot_product = sum(a * b for a, b in zip(gen_vec, exp_vec))
        norm_gen = math.sqrt(sum(a * a for a in gen_vec))
        norm_exp = math.sqrt(sum(b * b for b in exp_vec))
        
        if norm_gen == 0 or norm_exp == 0:
            cosine = 0.0
        else:
            cosine = dot_product / (norm_gen * norm_exp)
        
        # Combine metrics
        correctness = 0.5 * jaccard + 0.5 * cosine
        
        return correctness
    
    def calculate_context_precision(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        answer: str
    ) -> float:
        """
        Context Precision
        
        Proportion of retrieved context that was useful for the answer.
        """
        if not retrieved_chunks:
            return 0.0
        
        useful_count = 0
        answer_lower = answer.lower()
        
        for chunk in retrieved_chunks:
            chunk_words = set(self._tokenize(chunk.lower()))
            answer_words = set(self._tokenize(answer_lower))
            
            # Check if chunk contributed to answer
            overlap = chunk_words & answer_words
            if len(overlap) > 3:  # Threshold for "useful"
                useful_count += 1
        
        return useful_count / len(retrieved_chunks)
    
    def calculate_context_recall(
        self,
        retrieved_chunks: List[str],
        expected_answer: str
    ) -> float:
        """
        Context Recall
        
        How much of the expected answer is covered by retrieved context.
        """
        if not retrieved_chunks:
            return 0.0
        
        expected_keywords = set(self._tokenize(expected_answer.lower()))
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
                    'in', 'for', 'on', 'with', 'at', 'by', 'and', 'or', 'but'}
        expected_keywords -= stopwords
        
        if not expected_keywords:
            return 1.0
        
        # Collect all words from retrieved chunks
        context_words = set()
        for chunk in retrieved_chunks:
            context_words.update(self._tokenize(chunk.lower()))
        
        covered = len(expected_keywords & context_words)
        recall = covered / len(expected_keywords)
        
        return recall
    
    def calculate_end_to_end_metrics(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        latency_ms: float = 0.0,
        tokens_used: int = 0
    ) -> EndToEndMetrics:
        """Calculate all end-to-end metrics."""
        return EndToEndMetrics(
            answer_correctness=self.calculate_answer_correctness(
                generated_answer, expected_answer
            ),
            context_precision=self.calculate_context_precision(
                retrieved_chunks, relevant_chunks, generated_answer
            ),
            context_recall=self.calculate_context_recall(
                retrieved_chunks, expected_answer
            ),
            answer_similarity=self.calculate_bleu(generated_answer, expected_answer),
            latency_ms=latency_ms,
            tokens_used=tokens_used
        )
    
    # =========================================
    # HELPER METHODS
    # =========================================
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if not text:
            return []
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams from token list."""
        if len(tokens) < n:
            return []
        
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]