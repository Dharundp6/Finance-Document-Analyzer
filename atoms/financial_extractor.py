"""
Atomic component for extracting financial entities and metrics.
Identifies currencies, percentages, dates, and financial terms.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from config.logging_config import logger
from core.types import FinancialEntity


@dataclass
class ExtractionConfig:
    """Configuration for financial extraction."""
    extract_currencies: bool = True
    extract_percentages: bool = True
    extract_dates: bool = True
    extract_metrics: bool = True
    extract_ratios: bool = True
    context_window: int = 50  # Characters before/after for context


class FinancialExtractor:
    """
    Extracts financial entities from text.
    Identifies and normalizes financial data points.
    """
    
    # Currency patterns
    CURRENCY_PATTERNS = {
        'USD': [
            r'\$\s*([\d,]+(?:\.\d{1,2})?)\s*(?:million|billion|trillion|M|B|T)?',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:USD|dollars?)',
            r'US\$\s*([\d,]+(?:\.\d{1,2})?)'
        ],
        'EUR': [
            r'€\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:EUR|euros?)'
        ],
        'GBP': [
            r'£\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:GBP|pounds?)'
        ],
        'INR': [
            r'₹\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:INR|rupees?)',
            r'Rs\.?\s*([\d,]+(?:\.\d{1,2})?)'
        ]
    }
    
    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(
        r'(-?[\d,]+(?:\.\d+)?)\s*%'
    )
    
    # Date patterns
    DATE_PATTERNS = [
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', '%m/%d/%Y'),
        (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', '%Y/%m/%d'),
        (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})', '%b %d %Y'),
        (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})', '%d %b %Y'),
        (r'(Q[1-4])\s+(\d{4})', None),  # Fiscal quarters
        (r'FY\s*(\d{4})', None),  # Fiscal years
    ]
    
    # Financial metrics and terms
    FINANCIAL_METRICS = {
        'revenue': ['revenue', 'sales', 'turnover', 'top line'],
        'profit': ['profit', 'earnings', 'income', 'bottom line'],
        'margin': ['margin', 'gross margin', 'operating margin', 'net margin'],
        'growth': ['growth', 'yoy', 'year-over-year', 'cagr'],
        'ratio': ['ratio', 'p/e', 'debt-to-equity', 'current ratio', 'roi', 'roe'],
        'expense': ['expense', 'cost', 'expenditure', 'opex', 'capex'],
        'asset': ['asset', 'total assets', 'current assets', 'fixed assets'],
        'liability': ['liability', 'debt', 'obligation', 'payable'],
        'equity': ['equity', 'shareholder equity', 'book value'],
        'cash_flow': ['cash flow', 'operating cash', 'free cash flow', 'fcf']
    }
    
    # Multipliers for values
    MULTIPLIERS = {
        'trillion': 1e12, 't': 1e12,
        'billion': 1e9, 'b': 1e9, 'bn': 1e9,
        'million': 1e6, 'm': 1e6, 'mn': 1e6, 'mm': 1e6,
        'thousand': 1e3, 'k': 1e3,
        'lakh': 1e5, 'lac': 1e5,
        'crore': 1e7, 'cr': 1e7
    }
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the financial extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()
        
        # Compile currency patterns
        self.compiled_currency_patterns = {}
        for currency, patterns in self.CURRENCY_PATTERNS.items():
            self.compiled_currency_patterns[currency] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def extract_all(self, text: str) -> List[FinancialEntity]:
        """
        Extract all financial entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted financial entities
        """
        entities = []
        
        if self.config.extract_currencies:
            entities.extend(self.extract_currencies(text))
        
        if self.config.extract_percentages:
            entities.extend(self.extract_percentages(text))
        
        if self.config.extract_dates:
            entities.extend(self.extract_dates(text))
        
        if self.config.extract_metrics:
            entities.extend(self.extract_financial_metrics(text))
        
        return entities
    
    def extract_currencies(self, text: str) -> List[FinancialEntity]:
        """Extract currency values from text."""
        entities = []
        
        for currency, patterns in self.compiled_currency_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value_str = match.group(1) if match.groups() else match.group()
                    normalized = self._normalize_number(value_str)
                    
                    # Check for multiplier
                    full_match = match.group()
                    multiplier = self._detect_multiplier(full_match)
                    if multiplier and normalized:
                        normalized *= multiplier
                    
                    context = self._get_context(text, match.start(), match.end())
                    
                    entities.append(FinancialEntity(
                        entity_type=f"currency_{currency}",
                        value=full_match,
                        normalized_value=normalized,
                        context=context,
                        confidence=0.9
                    ))
        
        return entities
    
    def extract_percentages(self, text: str) -> List[FinancialEntity]:
        """Extract percentage values from text."""
        entities = []
        
        for match in self.PERCENTAGE_PATTERN.finditer(text):
            value_str = match.group(1)
            normalized = self._normalize_number(value_str)
            context = self._get_context(text, match.start(), match.end())
            
            # Determine if it's a growth rate, margin, or other
            percentage_type = self._classify_percentage(context)
            
            entities.append(FinancialEntity(
                entity_type=f"percentage_{percentage_type}",
                value=match.group(),
                normalized_value=normalized,
                context=context,
                confidence=0.95
            ))
        
        return entities
    
    def extract_dates(self, text: str) -> List[FinancialEntity]:
        """Extract dates and fiscal periods from text."""
        entities = []
        
        for pattern, date_format in self.DATE_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            
            for match in compiled.finditer(text):
                value = match.group()
                normalized = None
                
                # Try to parse the date
                if date_format:
                    try:
                        # Reconstruct date string
                        groups = match.groups()
                        if len(groups) == 3:
                            date_str = ' '.join(str(g) for g in groups)
                            normalized = datetime.strptime(date_str, date_format)
                    except ValueError:
                        pass
                
                context = self._get_context(text, match.start(), match.end())
                
                entities.append(FinancialEntity(
                    entity_type="date",
                    value=value,
                    normalized_value=normalized,
                    context=context,
                    confidence=0.85
                ))
        
        return entities
    
    def extract_financial_metrics(self, text: str) -> List[FinancialEntity]:
        """Extract mentions of financial metrics."""
        entities = []
        text_lower = text.lower()
        
        for metric_type, keywords in self.FINANCIAL_METRICS.items():
            for keyword in keywords:
                pattern = re.compile(
                    rf'\b{re.escape(keyword)}\b',
                    re.IGNORECASE
                )
                
                for match in pattern.finditer(text):
                    context = self._get_context(text, match.start(), match.end())
                    
                    # Look for associated numbers
                    associated_value = self._find_associated_number(text, match.end())
                    
                    entities.append(FinancialEntity(
                        entity_type=f"metric_{metric_type}",
                        value=match.group(),
                        normalized_value=associated_value,
                        context=context,
                        confidence=0.8
                    ))
        
        return entities
    
    def _normalize_number(self, value_str: str) -> Optional[float]:
        """Normalize a number string to float."""
        try:
            # Remove commas and spaces
            cleaned = value_str.replace(',', '').replace(' ', '').strip()
            
            # Handle parentheses for negative numbers
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            return float(cleaned)
        except (ValueError, InvalidOperation):
            return None
    
    def _detect_multiplier(self, text: str) -> Optional[float]:
        """Detect multiplier keywords in text."""
        text_lower = text.lower()
        
        for keyword, multiplier in self.MULTIPLIERS.items():
            if keyword in text_lower:
                return multiplier
        
        return None
    
    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get surrounding context for a match."""
        context_start = max(0, start - self.config.context_window)
        context_end = min(len(text), end + self.config.context_window)
        
        context = text[context_start:context_end].strip()
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = '...' + context
        if context_end < len(text):
            context = context + '...'
        
        return context
    
    def _classify_percentage(self, context: str) -> str:
        """Classify the type of percentage based on context."""
        context_lower = context.lower()
        
        if any(kw in context_lower for kw in ['growth', 'increase', 'decrease', 'change', 'yoy']):
            return 'growth'
        elif any(kw in context_lower for kw in ['margin', 'gross', 'operating', 'net']):
            return 'margin'
        elif any(kw in context_lower for kw in ['rate', 'interest', 'discount']):
            return 'rate'
        elif any(kw in context_lower for kw in ['share', 'market', 'ownership']):
            return 'share'
        else:
            return 'other'
    
    def _find_associated_number(self, text: str, position: int) -> Optional[float]:
        """Find a number associated with a metric mention."""
        # Look in the next 100 characters
        search_text = text[position:position + 100]
        
        # Look for currency or number
        number_pattern = re.compile(r'[\$€£₹]?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?')
        match = number_pattern.search(search_text)
        
        if match:
            return self._normalize_number(match.group(1))
        
        return None


def extract_financial_entities(text: str, **kwargs) -> List[FinancialEntity]:
    """Convenience function to extract financial entities."""
    config = ExtractionConfig(**kwargs)
    extractor = FinancialExtractor(config)
    return extractor.extract_all(text)