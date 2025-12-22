"""
Query analyzer for understanding user intent and query type.
Analyzes queries before generating responses.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config.logging_config import logger


class QueryType(Enum):
    """Types of queries."""
    GREETING = "greeting"             # Hi, Hello, Hey
    CHITCHAT = "chitchat"             # How are you? What's up?
    GRATITUDE = "gratitude"           # Thank you, Thanks
    FAREWELL = "farewell"             # Bye, Goodbye
    HELP = "help"                     # Help, What can you do?
    UNCLEAR = "unclear"               # Too short or unclear
    FACTUAL = "factual"               # What is X? How much is Y?
    COMPARISON = "comparison"         # Compare X and Y
    TREND = "trend"                   # How has X changed over time?
    AGGREGATION = "aggregation"       # Total, sum, average
    LIST = "list"                     # List all X, what are the X?
    EXPLANATION = "explanation"       # Why? Explain? How does?
    YES_NO = "yes_no"                 # Is X? Does Y? Are there?
    SUMMARY = "summary"               # Summarize, overview, brief


class ResponseFormat(Enum):
    """How to format the response."""
    DIRECT_MESSAGE = "direct_message"     # Simple text response
    SINGLE_VALUE = "single_value"         # Just one number/fact
    BULLET_LIST = "bullet_list"           # List of items
    TABLE = "table"                       # Tabular data
    PARAGRAPH = "paragraph"               # Narrative explanation
    COMPARISON_TABLE = "comparison"       # Side-by-side comparison
    STEP_BY_STEP = "step_by_step"         # Sequential explanation


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    query_type: QueryType
    response_format: ResponseFormat
    requires_document_search: bool = True
    key_entities: List[str] = field(default_factory=list)
    time_references: List[str] = field(default_factory=list)
    metrics_requested: List[str] = field(default_factory=list)
    comparison_items: List[str] = field(default_factory=list)
    is_specific: bool = True
    requires_calculation: bool = False
    confidence: float = 1.0
    direct_response: Optional[str] = None  # For greetings, chitchat, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "response_format": self.response_format.value,
            "requires_document_search": self.requires_document_search,
            "key_entities": self.key_entities,
            "metrics_requested": self.metrics_requested,
            "is_specific": self.is_specific
        }


class QueryAnalyzer:
    """
    Analyzes queries to understand intent and determine response format.
    First checks for non-document queries (greetings, chitchat, etc.)
    """
    
    # Non-document queries - these don't need document search
    GREETING_PATTERNS = [
        r'^hi$', r'^hello$', r'^hey$', r'^hi there$', r'^hello there$',
        r'^good\s*(morning|afternoon|evening)$', r'^greetings$',
        r'^howdy$', r'^hiya$', r'^yo$'
    ]
    
    CHITCHAT_PATTERNS = [
        r'^how\s+are\s+you', r"^what'?s\s+up", r'^how\s+do\s+you\s+do',
        r"^how'?s\s+it\s+going", r'^what\s+is\s+your\s+name',
        r'^who\s+are\s+you', r'^are\s+you\s+(a\s+)?(robot|ai|bot)',
        r'^what\s+can\s+you\s+do'
    ]
    
    GRATITUDE_PATTERNS = [
        r'^thank\s*you', r'^thanks', r'^thx', r'^appreciate',
        r'^great\s+job', r'^well\s+done', r'^good\s+job'
    ]
    
    FAREWELL_PATTERNS = [
        r'^bye', r'^goodbye', r'^see\s+you', r'^take\s+care',
        r'^later$', r'^cya$'
    ]
    
    HELP_PATTERNS = [
        r'^help$', r'^help\s+me', r'^what\s+can\s+you\s+(do|help)',
        r'^how\s+to\s+use', r'^commands$', r'^\?$'
    ]
    
    # Document-related query patterns
    QUERY_PATTERNS = {
        QueryType.COMPARISON: [
            r'\bcompare\b', r'\bversus\b', r'\bvs\.?\b', r'\bdifference\s+between\b',
            r'\bhigher\s+than\b', r'\blower\s+than\b', r'\bmore\s+than\b',
            r'\bless\s+than\b', r'\bbetter\b', r'\bworse\b'
        ],
        QueryType.TREND: [
            r'\btrend\b', r'\bover\s+time\b', r'\bover\s+the\s+years?\b',
            r'\bgrowth\b', r'\bchange[ds]?\b', r'\bincrease[ds]?\b', 
            r'\bdecrease[ds]?\b', r'\bevolution\b', r'\bhistor(y|ical)\b'
        ],
        QueryType.AGGREGATION: [
            r'\btotal\b', r'\bsum\b', r'\baverage\b', r'\bmean\b',
            r'\boverall\b', r'\bcombined\b', r'\baggregate\b'
        ],
        QueryType.LIST: [
            r'\blist\b', r'\bwhat\s+are\s+(the|all)\b', r'\bshow\s+(me\s+)?(all|the)\b',
            r'\benumerate\b', r'\bname\s+(all|the)\b', r'\bidentify\s+(all|the)\b'
        ],
        QueryType.EXPLANATION: [
            r'\bwhy\b', r'\bexplain\b', r'\bhow\s+does\b', r'\bhow\s+do\b',
            r'\breason\b', r'\bcause\b', r'\bimpact\b', r'\baffect\b'
        ],
        QueryType.YES_NO: [
            r'^is\s+', r'^are\s+', r'^does\s+', r'^do\s+', r'^did\s+',
            r'^was\s+', r'^were\s+', r'^has\s+', r'^have\s+', r'^can\s+'
        ],
        QueryType.SUMMARY: [
            r'\bsummar(y|ize)\b', r'\boverview\b', r'\bbrief\b',
            r'\bhighlight\b', r'\bkey\s+points?\b', r'\bmain\s+points?\b'
        ]
    }
    
    # Financial metrics patterns
    FINANCIAL_METRICS = [
        'revenue', 'sales', 'income', 'profit', 'loss', 'margin',
        'expense', 'cost', 'ebitda', 'ebit', 'eps', 'earnings',
        'asset', 'liability', 'equity', 'debt', 'cash', 'capital',
        'roi', 'roe', 'roa', 'ratio', 'growth', 'return', 'dividend',
        'share', 'stock', 'market', 'price', 'value'
    ]
    
    # Time reference patterns
    TIME_PATTERNS = [
        r'\b(q[1-4])\b', r'\b(fy\s*\d{2,4})\b', r'\b(20\d{2})\b',
        r'\b(fiscal\s+year)\b', r'\b(quarter)\b', r'\b(annual)\b',
        r'\b(yearly)\b', r'\b(monthly)\b', r'\b(ytd)\b'
    ]
    
    # Direct responses for non-document queries
    DIRECT_RESPONSES = {
        QueryType.GREETING: """👋 Hello! I'm your Financial Document Assistant.

I can help you analyze financial documents and answer questions about:
• Revenue, profit, and expenses
• Financial metrics and ratios
• Business segments and performance
• Trends and comparisons

**To get started:**
1. Upload a financial document (PDF, Excel, Word, CSV)
2. Ask me any question about its contents

What would you like to know?""",

        QueryType.CHITCHAT: """I'm doing great, thank you for asking! 😊

I'm a Financial Document Q&A Assistant powered by AI. I can help you:
• Analyze financial reports and documents
• Answer questions about financial data
• Compare metrics and find trends
• Summarize key financial information

Do you have any financial documents you'd like to explore?""",

        QueryType.GRATITUDE: """You're welcome! 😊 

I'm happy I could help. Feel free to ask more questions about your financial documents anytime.

Is there anything else you'd like to know?""",

        QueryType.FAREWELL: """Goodbye! 👋 

Thank you for using the Financial Document Assistant. Have a great day!

Come back anytime you need help analyzing financial documents.""",

        QueryType.HELP: """📚 **Financial Document Q&A - Help Guide**

**What I Can Do:**
• Answer questions about uploaded financial documents
• Extract specific metrics (revenue, profit, expenses, etc.)
• Compare data across time periods
• Summarize financial information
• List key items and highlights

**How to Use:**
1. **Upload a document:** `upload /path/to/file.pdf`
2. **Ask questions:** Just type your question naturally

**Example Questions:**
• "What is the total revenue?"
• "Compare Q1 and Q2 performance"
• "List all business segments"
• "What is the profit margin?"
• "Summarize the financial highlights"

**Commands:**
• `docs` - List uploaded documents
• `stats` - Show system statistics
• `help` - Show this help message
• `quit` - Exit the program

What would you like to know?""",

        QueryType.UNCLEAR: """🤔 I'm not sure I understand your question.

Could you please be more specific? Here are some examples of questions I can answer:

• "What is the total revenue for 2023?"
• "What are the main business segments?"
• "Compare expenses between Q1 and Q2"
• "List all risk factors mentioned"

Or type `help` for more information."""
    }
    
    # Minimum query length for document search
    MIN_QUERY_LENGTH = 3
    MIN_MEANINGFUL_WORDS = 2
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to understand intent.
        
        Args:
            query: User's query string
            
        Returns:
            QueryAnalysis with detailed analysis
        """
        original_query = query
        query_lower = query.lower().strip()
        
        logger.debug(f"Analyzing query: '{query_lower}'")
        
        # Step 1: Check for non-document queries first
        non_doc_result = self._check_non_document_query(query_lower)
        if non_doc_result:
            logger.info(f"Non-document query detected: {non_doc_result.query_type.value}")
            non_doc_result.original_query = original_query
            return non_doc_result
        
        # Step 2: Check if query is too short or unclear
        if self._is_unclear_query(query_lower):
            logger.info("Query is unclear or too short")
            return QueryAnalysis(
                original_query=original_query,
                query_type=QueryType.UNCLEAR,
                response_format=ResponseFormat.DIRECT_MESSAGE,
                requires_document_search=False,
                direct_response=self.DIRECT_RESPONSES[QueryType.UNCLEAR]
            )
        
        # Step 3: This is a document-related query
        query_type = self._detect_query_type(query_lower)
        response_format = self._determine_response_format(query_type, query_lower)
        
        # Extract various entities
        key_entities = self._extract_entities(query_lower)
        time_references = self._extract_time_references(query_lower)
        metrics = self._extract_metrics(query_lower)
        comparison_items = self._extract_comparison_items(query_lower) if query_type == QueryType.COMPARISON else []
        
        is_specific = self._is_specific_query(query_lower, metrics, key_entities)
        requires_calculation = self._requires_calculation(query_lower)
        
        analysis = QueryAnalysis(
            original_query=original_query,
            query_type=query_type,
            response_format=response_format,
            requires_document_search=True,
            key_entities=key_entities,
            time_references=time_references,
            metrics_requested=metrics,
            comparison_items=comparison_items,
            is_specific=is_specific,
            requires_calculation=requires_calculation
        )
        
        logger.info(f"Document query: type={query_type.value}, metrics={metrics}")
        return analysis
    
    def _check_non_document_query(self, query: str) -> Optional[QueryAnalysis]:
        """Check if this is a greeting, chitchat, or other non-document query."""
        
        # Check greetings
        for pattern in self.GREETING_PATTERNS:
            if re.match(pattern, query, re.IGNORECASE):
                return QueryAnalysis(
                    original_query=query,
                    query_type=QueryType.GREETING,
                    response_format=ResponseFormat.DIRECT_MESSAGE,
                    requires_document_search=False,
                    direct_response=self.DIRECT_RESPONSES[QueryType.GREETING]
                )
        
        # Check chitchat
        for pattern in self.CHITCHAT_PATTERNS:
            if re.match(pattern, query, re.IGNORECASE):
                return QueryAnalysis(
                    original_query=query,
                    query_type=QueryType.CHITCHAT,
                    response_format=ResponseFormat.DIRECT_MESSAGE,
                    requires_document_search=False,
                    direct_response=self.DIRECT_RESPONSES[QueryType.CHITCHAT]
                )
        
        # Check gratitude
        for pattern in self.GRATITUDE_PATTERNS:
            if re.match(pattern, query, re.IGNORECASE):
                return QueryAnalysis(
                    original_query=query,
                    query_type=QueryType.GRATITUDE,
                    response_format=ResponseFormat.DIRECT_MESSAGE,
                    requires_document_search=False,
                    direct_response=self.DIRECT_RESPONSES[QueryType.GRATITUDE]
                )
        
        # Check farewell
        for pattern in self.FAREWELL_PATTERNS:
            if re.match(pattern, query, re.IGNORECASE):
                return QueryAnalysis(
                    original_query=query,
                    query_type=QueryType.FAREWELL,
                    response_format=ResponseFormat.DIRECT_MESSAGE,
                    requires_document_search=False,
                    direct_response=self.DIRECT_RESPONSES[QueryType.FAREWELL]
                )
        
        # Check help requests
        for pattern in self.HELP_PATTERNS:
            if re.match(pattern, query, re.IGNORECASE):
                return QueryAnalysis(
                    original_query=query,
                    query_type=QueryType.HELP,
                    response_format=ResponseFormat.DIRECT_MESSAGE,
                    requires_document_search=False,
                    direct_response=self.DIRECT_RESPONSES[QueryType.HELP]
                )
        
        return None
    
    def _is_unclear_query(self, query: str) -> bool:
        """Check if query is too short or unclear."""
        
        # Too short
        if len(query) < self.MIN_QUERY_LENGTH:
            return True
        
        # Just punctuation or special characters
        if not re.search(r'[a-zA-Z]', query):
            return True
        
        # Single word that's not a financial term
        words = query.split()
        if len(words) < self.MIN_MEANINGFUL_WORDS:
            # Allow single financial terms
            if not any(metric in query for metric in self.FINANCIAL_METRICS):
                return True
        
        return False
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of document query."""
        for query_type, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        
        return QueryType.FACTUAL
    
    def _determine_response_format(self, query_type: QueryType, query: str) -> ResponseFormat:
        """Determine the best response format."""
        format_map = {
            QueryType.FACTUAL: ResponseFormat.SINGLE_VALUE,
            QueryType.COMPARISON: ResponseFormat.COMPARISON_TABLE,
            QueryType.TREND: ResponseFormat.PARAGRAPH,
            QueryType.AGGREGATION: ResponseFormat.SINGLE_VALUE,
            QueryType.LIST: ResponseFormat.BULLET_LIST,
            QueryType.EXPLANATION: ResponseFormat.PARAGRAPH,
            QueryType.YES_NO: ResponseFormat.PARAGRAPH,
            QueryType.SUMMARY: ResponseFormat.BULLET_LIST
        }
        
        base_format = format_map.get(query_type, ResponseFormat.PARAGRAPH)
        
        if 'breakdown' in query or 'by segment' in query or 'by category' in query:
            return ResponseFormat.TABLE
        
        if 'step' in query or 'how to' in query:
            return ResponseFormat.STEP_BY_STEP
        
        return base_format
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query."""
        entities = []
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        return entities
    
    def _extract_time_references(self, query: str) -> List[str]:
        """Extract time references."""
        references = []
        for pattern in self.TIME_PATTERNS:
            matches = re.findall(pattern, query, re.IGNORECASE)
            references.extend(matches)
        return references
    
    def _extract_metrics(self, query: str) -> List[str]:
        """Extract financial metrics mentioned."""
        metrics = []
        for metric in self.FINANCIAL_METRICS:
            if metric in query:
                metrics.append(metric)
        return metrics
    
    def _extract_comparison_items(self, query: str) -> List[str]:
        """Extract items being compared."""
        items = []
        patterns = [
            r'(\w+)\s+(?:vs\.?|versus)\s+(\w+)',
            r'compare\s+(\w+)\s+(?:and|to|with)\s+(\w+)',
            r'between\s+(\w+)\s+and\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                items.extend(match.groups())
        
        return items
    
    def _is_specific_query(self, query: str, metrics: List[str], entities: List[str]) -> bool:
        """Check if query is specific or general."""
        if metrics or entities:
            return True
        
        general_patterns = [
            r'\btell\s+me\s+about\b', r'\bwhat\s+do\s+you\s+know\b',
            r'\bgive\s+me\s+information\b', r'\boverall\b'
        ]
        
        for pattern in general_patterns:
            if re.search(pattern, query):
                return False
        
        return True
    
    def _requires_calculation(self, query: str) -> bool:
        """Check if query requires calculation."""
        calc_patterns = [
            r'\bcalculate\b', r'\bcompute\b', r'\bsum\b', r'\btotal\b',
            r'\baverage\b', r'\bpercentage\b', r'\bratio\b', r'\bgrowth\s+rate\b'
        ]
        
        for pattern in calc_patterns:
            if re.search(pattern, query):
                return True
        
        return False


# Global instance
_query_analyzer: Optional[QueryAnalyzer] = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get the global query analyzer."""
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer


def analyze_query(query: str) -> QueryAnalysis:
    """Convenience function to analyze a query."""
    return get_query_analyzer().analyze(query)