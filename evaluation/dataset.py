"""
Evaluation dataset management.
Create, load, and manage test datasets for RAG evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

from config.logging_config import logger


@dataclass
class TestCase:
    """Single test case for evaluation."""
    id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevant_keywords: List[str] = field(default_factory=list)
    query_type: str = "factual"
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        return cls(**data)


@dataclass
class EvaluationDataset:
    """Collection of test cases for evaluation."""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the dataset."""
        self.test_cases.append(test_case)
    
    def add_test(
        self,
        query: str,
        expected_answer: str,
        relevant_keywords: List[str] = None,
        query_type: str = "factual",
        difficulty: str = "medium"
    ) -> TestCase:
        """Convenience method to add a test case."""
        test_case = TestCase(
            id=str(uuid.uuid4())[:8],
            query=query,
            expected_answer=expected_answer,
            relevant_keywords=relevant_keywords or [],
            query_type=query_type,
            difficulty=difficulty
        )
        self.test_cases.append(test_case)
        return test_case
    
    def save(self, path: str) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "test_cases": [tc.to_dict() for tc in self.test_cases]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EvaluationDataset':
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = [TestCase.from_dict(tc) for tc in data.get('test_cases', [])]
        
        dataset = cls(
            name=data['name'],
            description=data['description'],
            test_cases=test_cases,
            created_at=data.get('created_at', ''),
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {})
        )
        
        logger.info(f"Loaded dataset with {len(test_cases)} test cases")
        return dataset
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __iter__(self):
        return iter(self.test_cases)


def create_nvidia_evaluation_dataset() -> EvaluationDataset:
    """
    Create evaluation dataset based on NVIDIA Q1 FY2024 Financial Results.
    """
    dataset = EvaluationDataset(
        name="NVIDIA Q1 FY2024 Financial RAG Evaluation",
        description="Test cases for evaluating NVIDIA financial document Q&A from Q1 FY2024 earnings report",
        metadata={
            "source": "NVIDIA Q1 FY2024 Earnings Report",
            "fiscal_period": "Q1 FY2024",
            "report_date": "April 30, 2023"
        }
    )
    
    # ==================== FACTUAL QUESTIONS - EASY ====================
    
    dataset.add_test(
        query="What was NVIDIA's revenue in Q1 FY2024?",
        expected_answer="NVIDIA's revenue in Q1 FY2024 was $7.19 billion (or $7,192 million).",
        relevant_keywords=["revenue", "Q1", "FY2024", "7.19", "billion"],
        query_type="factual",
        difficulty="easy"
    )
    
    dataset.add_test(
        query="What was the GAAP earnings per diluted share?",
        expected_answer="GAAP earnings per diluted share were $0.82 for Q1 FY2024.",
        relevant_keywords=["GAAP", "earnings", "diluted", "share", "0.82"],
        query_type="factual",
        difficulty="easy"
    )
    
    dataset.add_test(
        query="What was the Data Center revenue in Q1?",
        expected_answer="Data Center revenue was a record $4.28 billion in Q1 FY2024.",
        relevant_keywords=["data center", "revenue", "4.28", "billion", "record"],
        query_type="factual",
        difficulty="easy"
    )
    
    dataset.add_test(
        query="What was the Gaming revenue?",
        expected_answer="Gaming revenue was $2.24 billion in Q1 FY2024.",
        relevant_keywords=["gaming", "revenue", "2.24", "billion"],
        query_type="factual",
        difficulty="easy"
    )
    
    dataset.add_test(
        query="What is the revenue outlook for Q2 FY2024?",
        expected_answer="The revenue outlook for Q2 FY2024 is $11.00 billion, plus or minus 2%.",
        relevant_keywords=["Q2", "outlook", "11", "billion", "revenue"],
        query_type="factual",
        difficulty="easy"
    )
    
    # ==================== FACTUAL QUESTIONS - MEDIUM ====================
    
    dataset.add_test(
        query="What was the GAAP gross margin in Q1?",
        expected_answer="The GAAP gross margin was 64.6% in Q1 FY2024.",
        relevant_keywords=["GAAP", "gross", "margin", "64.6", "percent"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What was the non-GAAP gross margin?",
        expected_answer="The non-GAAP gross margin was 66.8% in Q1 FY2024.",
        relevant_keywords=["non-GAAP", "gross", "margin", "66.8", "percent"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What was the GAAP net income?",
        expected_answer="GAAP net income was $2,043 million (or $2.043 billion) in Q1 FY2024.",
        relevant_keywords=["GAAP", "net", "income", "2043", "million"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What was the operating income in Q1?",
        expected_answer="GAAP operating income was $2,140 million in Q1 FY2024.",
        relevant_keywords=["operating", "income", "2140", "million"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What was the Professional Visualization revenue?",
        expected_answer="Professional Visualization revenue was $295 million in Q1 FY2024.",
        relevant_keywords=["professional", "visualization", "revenue", "295", "million"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What was the Automotive revenue?",
        expected_answer="Automotive revenue was a record $296 million in Q1 FY2024.",
        relevant_keywords=["automotive", "revenue", "296", "million", "record"],
        query_type="factual",
        difficulty="medium"
    )
    
    # ==================== COMPARISON QUESTIONS ====================
    
    dataset.add_test(
        query="How did Q1 FY2024 revenue compare to Q4 FY2023?",
        expected_answer="Q1 FY2024 revenue of $7.19 billion was up 19% from Q4 FY2023 revenue of $6.05 billion.",
        relevant_keywords=["Q1", "Q4", "revenue", "compare", "19%", "up"],
        query_type="comparison",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="Compare Q1 FY2024 revenue to Q1 FY2023",
        expected_answer="Q1 FY2024 revenue of $7.19 billion was down 13% from Q1 FY2023 revenue of $8.29 billion.",
        relevant_keywords=["Q1", "FY2024", "FY2023", "revenue", "down", "13%"],
        query_type="comparison",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="How did GAAP EPS change from previous quarter?",
        expected_answer="GAAP EPS increased 44% from $0.57 in Q4 FY2023 to $0.82 in Q1 FY2024.",
        relevant_keywords=["GAAP", "EPS", "earnings", "44%", "increase"],
        query_type="comparison",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="Compare GAAP vs non-GAAP gross margin in Q1",
        expected_answer="GAAP gross margin was 64.6% while non-GAAP gross margin was 66.8% in Q1 FY2024, a difference of 2.2 percentage points.",
        relevant_keywords=["GAAP", "non-GAAP", "gross", "margin", "compare"],
        query_type="comparison",
        difficulty="hard"
    )
    
    dataset.add_test(
        query="How did Gaming revenue perform year-over-year?",
        expected_answer="Gaming revenue of $2.24 billion was down 38% from a year ago but up 22% from the previous quarter.",
        relevant_keywords=["gaming", "revenue", "year-over-year", "down", "38%"],
        query_type="comparison",
        difficulty="medium"
    )
    
    # ==================== TREND/CALCULATION QUESTIONS ====================
    
    dataset.add_test(
        query="What was the percentage change in Data Center revenue from previous quarter?",
        expected_answer="Data Center revenue increased by 18% from the previous quarter.",
        relevant_keywords=["data center", "revenue", "increase", "18%", "quarter"],
        query_type="trend",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="Calculate the total operating expenses for Q1",
        expected_answer="Total GAAP operating expenses were $2,508 million in Q1 FY2024, consisting of R&D ($1,875M) and Sales, general and administrative ($633M).",
        relevant_keywords=["operating", "expenses", "2508", "million", "R&D"],
        query_type="calculation",
        difficulty="hard"
    )
    
    dataset.add_test(
        query="What was the free cash flow in Q1?",
        expected_answer="Free cash flow was $2,643 million in Q1 FY2024.",
        relevant_keywords=["free", "cash", "flow", "2643", "million"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="How much did NVIDIA return to shareholders in dividends?",
        expected_answer="NVIDIA returned $99 million in cash dividends to shareholders during Q1 FY2024.",
        relevant_keywords=["dividends", "shareholders", "99", "million"],
        query_type="factual",
        difficulty="easy"
    )
    
    # ==================== LIST/ENUMERATION QUESTIONS ====================
    
    dataset.add_test(
        query="What are NVIDIA's business segments?",
        expected_answer="NVIDIA's business segments are: Data Center, Gaming, Professional Visualization, and Automotive.",
        relevant_keywords=["business", "segments", "data center", "gaming", "automotive"],
        query_type="list",
        difficulty="easy"
    )
    
    dataset.add_test(
        query="What products are in NVIDIA's data center family?",
        expected_answer="NVIDIA's data center family includes: H100, Grace CPU, Grace Hopper Superchip, NVLink, Quantum 400 InfiniBand, and BlueField-3 DPU.",
        relevant_keywords=["data center", "products", "H100", "Grace", "NVLink"],
        query_type="list",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="Which cloud providers were mentioned as H100 partners?",
        expected_answer="The cloud providers offering NVIDIA H100 include: Amazon Web Services (AWS), Google Cloud, Microsoft Azure, and Oracle Cloud Infrastructure.",
        relevant_keywords=["cloud", "providers", "H100", "AWS", "Azure", "Google"],
        query_type="list",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What GPUs were announced in the Gaming segment?",
        expected_answer="NVIDIA announced the GeForce RTX 4060 family and launched the GeForce RTX 4070 GPU based on the Ada architecture.",
        relevant_keywords=["gaming", "GPU", "GeForce", "RTX", "4060", "4070"],
        query_type="list",
        difficulty="medium"
    )
    
    # ==================== COMPLEX/ANALYTICAL QUESTIONS ====================
    
    dataset.add_test(
        query="What did Jensen Huang say about industry transitions?",
        expected_answer="Jensen Huang stated that the computer industry is going through two simultaneous transitions: accelerated computing and generative AI. He mentioned that a trillion dollars of installed global data center infrastructure will transition from general purpose to accelerated computing.",
        relevant_keywords=["Jensen", "Huang", "transitions", "AI", "generative", "accelerated"],
        query_type="summary",
        difficulty="hard"
    )
    
    dataset.add_test(
        query="Summarize the Q2 FY2024 financial outlook",
        expected_answer="Q2 FY2024 outlook: Revenue expected at $11.00 billion (±2%), GAAP gross margin at 68.6%, non-GAAP gross margin at 70.0%, GAAP operating expenses at ~$2.71B, non-GAAP operating expenses at ~$1.90B, and tax rate at 14.0% (±1%).",
        relevant_keywords=["Q2", "outlook", "revenue", "11", "billion", "gross margin"],
        query_type="summary",
        difficulty="hard"
    )
    
    dataset.add_test(
        query="What was NVIDIA's automotive design win pipeline?",
        expected_answer="NVIDIA's automotive design win pipeline has grown to $14 billion over the next six years, up from $11 billion a year ago.",
        relevant_keywords=["automotive", "pipeline", "14", "billion", "design", "win"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="Analyze NVIDIA's cash position",
        expected_answer="NVIDIA's cash, cash equivalents and marketable securities totaled $15,320 million as of April 30, 2023, up from $13,296 million at the end of Q4 FY2023.",
        relevant_keywords=["cash", "marketable", "securities", "15320", "million"],
        query_type="factual",
        difficulty="hard"
    )
    
    dataset.add_test(
        query="What was the year-over-year change in Professional Visualization revenue?",
        expected_answer="Professional Visualization revenue was down 53% year-over-year but up 31% from the previous quarter.",
        relevant_keywords=["professional", "visualization", "down", "53%", "year-over-year"],
        query_type="comparison",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="What is NVIDIA's next dividend payment?",
        expected_answer="NVIDIA will pay a quarterly cash dividend of $0.04 per share on June 30, 2023, to shareholders of record on June 8, 2023.",
        relevant_keywords=["dividend", "0.04", "June", "30", "2023"],
        query_type="factual",
        difficulty="easy"
    )
    
    # ==================== SPECIFIC ANNOUNCEMENTS/PARTNERSHIPS ====================
    
    dataset.add_test(
        query="What collaboration did NVIDIA announce with Microsoft?",
        expected_answer="NVIDIA announced multiple collaborations with Microsoft: integrating NVIDIA AI Enterprise software into Azure Machine Learning, connecting Microsoft 365 applications with Omniverse, and offering NVIDIA Omniverse Cloud as a fully managed service in Microsoft Azure.",
        relevant_keywords=["Microsoft", "collaboration", "Azure", "Omniverse", "AI"],
        query_type="list",
        difficulty="hard"
    )
    
    dataset.add_test(
        query="What was announced about BYD?",
        expected_answer="BYD, the world's leading electric vehicle maker, will extend its use of NVIDIA DRIVE Orin across new models.",
        relevant_keywords=["BYD", "electric", "vehicle", "DRIVE", "Orin"],
        query_type="factual",
        difficulty="medium"
    )
    
    dataset.add_test(
        query="How many DLSS gaming titles are available?",
        expected_answer="NVIDIA added 36 DLSS gaming titles, bringing the total number of games and apps to 300.",
        relevant_keywords=["DLSS", "gaming", "titles", "300", "apps"],
        query_type="factual",
        difficulty="easy"
    )
    
    dataset.add_test(
        query="What is NVIDIA AI Foundations?",
        expected_answer="NVIDIA AI Foundations is a service to help businesses create and operate custom large language models and generative AI models trained with their own proprietary data for domain-specific tasks.",
        relevant_keywords=["AI", "Foundations", "LLM", "generative", "custom"],
        query_type="factual",
        difficulty="medium"
    )
    
    return dataset


def create_financial_evaluation_dataset() -> EvaluationDataset:
    """
    Alias for backward compatibility.
    Returns the NVIDIA evaluation dataset.
    """
    return create_nvidia_evaluation_dataset()


# Example usage
if __name__ == "__main__":
    # Create the dataset
    dataset = create_nvidia_evaluation_dataset()
    
    # Save to file
    dataset.save("nvidia_q1_fy2024_evaluation.json")
    
    print(f"Created evaluation dataset with {len(dataset)} test cases")
    print("\nTest case distribution:")
    
    # Count by query type
    query_types = {}
    for test in dataset:
        query_types[test.query_type] = query_types.get(test.query_type, 0) + 1
    
    for qtype, count in query_types.items():
        print(f"  {qtype}: {count}")
    
    # Count by difficulty
    print("\nDifficulty distribution:")
    difficulties = {}
    for test in dataset:
        difficulties[test.difficulty] = difficulties.get(test.difficulty, 0) + 1
    
    for diff, count in difficulties.items():
        print(f"  {diff}: {count}")