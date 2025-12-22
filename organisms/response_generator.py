"""
Enhanced response generation with query analysis and structured formatting.
Uses Google Gemini Flash for fast, accurate responses.
"""

import time
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import uuid

import google.generativeai as genai

from config.settings import settings
from config.logging_config import logger
from core.types import GeneratedResponse, RetrievalResult, QueryContext
from core.exceptions import GenerationError
from templates.prompts import PromptTemplates
from atoms.query_analyzer import (
    QueryAnalyzer, QueryAnalysis, QueryType, ResponseFormat,
    get_query_analyzer
)


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    model: str = "gemini-flash-latest"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40


class ResponseGenerator:
    """
    Enhanced response generator with query analysis.
    Handles both document queries and conversational queries.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize the response generator."""
        self.config = config or GenerationConfig(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens
        )
        
        # Initialize Gemini
        genai.configure(api_key=settings.google_api_key)
        
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        self.prompt_templates = PromptTemplates()
        self.query_analyzer = get_query_analyzer()
        
        logger.info(f"ResponseGenerator initialized with model: {self.config.model}")
    
    def generate(
        self,
        query: str,
        context: str,
        query_context: Optional[QueryContext] = None,
        retrieval_results: Optional[List[RetrievalResult]] = None,
        max_retries: int = 3
    ) -> GeneratedResponse:
        """
        Generate a response to the query.
        First analyzes the query, then responds appropriately.
        """
        start_time = time.time()
        response_id = str(uuid.uuid4())
        query_id = query_context.query_id if query_context else str(uuid.uuid4())
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Step 1: Analyze the query
        analysis = self.query_analyzer.analyze(query)
        logger.info(f"Query type: {analysis.query_type.value}")
        
        # Step 2: Handle non-document queries (greetings, help, etc.)
        if not analysis.requires_document_search:
            logger.info("Returning direct response (no document search needed)")
            return self._create_direct_response(
                response_id, query_id, analysis, time.time() - start_time
            )
        
        # Step 3: Check if we have context for document queries
        if not context or not context.strip():
            logger.info("No context available for document query")
            answer = self._generate_no_context_response(query)
            return self._create_response(
                response_id, query_id, answer, retrieval_results,
                0.0, time.time() - start_time, analysis
            )
        
        # Step 4: Generate response from documents
        prompt = self._build_prompt(query, context, analysis)
        answer = self._generate_with_retry(prompt, max_retries)
        answer = self._post_process_response(answer, analysis)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(retrieval_results, analysis)
        
        processing_time = time.time() - start_time
        
        return self._create_response(
            response_id, query_id, answer, retrieval_results,
            confidence, processing_time, analysis
        )
    
    def _create_direct_response(
        self,
        response_id: str,
        query_id: str,
        analysis: QueryAnalysis,
        processing_time: float
    ) -> GeneratedResponse:
        """Create response for non-document queries (greetings, help, etc.)."""
        return GeneratedResponse(
            response_id=response_id,
            query_id=query_id,
            answer=analysis.direct_response or "I'm here to help!",
            sources=[],
            confidence=1.0,
            processing_time=processing_time,
            token_usage={'prompt_tokens': 0, 'completion_tokens': 0},
            metadata={
                'query_type': analysis.query_type.value,
                'response_format': analysis.response_format.value,
                'requires_document_search': False
            }
        )
    
    def _build_prompt(self, query: str, context: str, analysis: QueryAnalysis) -> str:
        """Build the appropriate prompt based on query analysis."""
        type_to_template = {
            QueryType.FACTUAL: 'factual',
            QueryType.COMPARISON: 'comparison',
            QueryType.TREND: 'trend',
            QueryType.AGGREGATION: 'aggregation',
            QueryType.LIST: 'list',
            QueryType.EXPLANATION: 'explanation',
            QueryType.YES_NO: 'yes_no',
            QueryType.SUMMARY: 'summary'
        }
        
        template_key = type_to_template.get(analysis.query_type, 'default')
        template = self.prompt_templates.get_prompt(template_key)
        
        return template.format(context=context, query=query)
    
    def _generate_with_retry(self, prompt: str, max_retries: int) -> str:
        """Generate response with retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.parts:
                    return response.text
                else:
                    return "I couldn't generate a response. Please try rephrasing your question."
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                if "ResourceExhausted" in error_str or "429" in error_str:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Generation error (attempt {attempt + 1}): {e}")
                    time.sleep(5)
        
        raise GenerationError(f"Failed to generate response: {last_error}")
    
    def _generate_no_context_response(self, query: str) -> str:
        """Generate response when no context is available."""
        return f"""**No Relevant Information Found**

I searched the uploaded documents but couldn't find information about: "{query}"

**Possible Reasons:**
• No documents have been uploaded yet
• The uploaded documents don't contain this information
• Try using different keywords

**Suggestions:**
• Upload relevant financial documents first
• Ask about specific metrics: revenue, profit, expenses, margins
• Use the `docs` command to see uploaded documents
• Use the `help` command for more guidance"""
    
    def _post_process_response(self, answer: str, analysis: QueryAnalysis) -> str:
        """Post-process and clean up the response."""
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        return answer.strip()
    
    def _calculate_confidence(
        self,
        retrieval_results: Optional[List[RetrievalResult]],
        analysis: QueryAnalysis
    ) -> float:
        """Calculate confidence score."""
        if not retrieval_results:
            return 0.0
        
        scores = [r.score for r in retrieval_results[:5]]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if analysis.query_type == QueryType.FACTUAL and avg_score > 0.7:
            avg_score = min(avg_score * 1.1, 1.0)
        
        if analysis.query_type == QueryType.EXPLANATION:
            avg_score = avg_score * 0.9
        
        return min(max(avg_score, 0.0), 1.0)
    
    def _create_response(
        self,
        response_id: str,
        query_id: str,
        answer: str,
        retrieval_results: Optional[List[RetrievalResult]],
        confidence: float,
        processing_time: float,
        analysis: QueryAnalysis
    ) -> GeneratedResponse:
        """Create the final response object."""
        return GeneratedResponse(
            response_id=response_id,
            query_id=query_id,
            answer=answer,
            sources=retrieval_results or [],
            confidence=confidence,
            processing_time=processing_time,
            token_usage={
                'prompt_tokens': 0,
                'completion_tokens': len(answer) // 4
            },
            metadata={
                'model': self.config.model,
                'query_type': analysis.query_type.value,
                'response_format': analysis.response_format.value,
                'requires_document_search': analysis.requires_document_search,
                'metrics_found': analysis.metrics_requested
            }
        )
    
    def generate_with_streaming(
        self,
        query: str,
        context: str,
        query_context: Optional[QueryContext] = None
    ):
        """Generate response with streaming."""
        # Analyze first
        analysis = self.query_analyzer.analyze(query)
        
        # Handle non-document queries directly
        if not analysis.requires_document_search:
            yield analysis.direct_response
            return
        
        # Handle document queries
        if not context or not context.strip():
            yield self._generate_no_context_response(query)
            return
        
        prompt = self._build_prompt(query, context, analysis)
        
        try:
            response = self.model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise GenerationError(f"Streaming failed: {e}")