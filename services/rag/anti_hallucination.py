"""
Anti-Hallucination Components

This module contains specialized prompts and utilities designed to minimize
hallucinations and improve answer quality in the RAG system.
"""

class AntiHallucinationPrompts:
    """Prompts designed to minimize hallucinations and improve answer quality"""
    
    STRICT_GROUNDING_PROMPT = """You are a knowledgeable assistant providing comprehensive answers based on the provided sources.

GUIDELINES FOR COMPLETE ANSWERS:
1. Provide a thorough and complete response using the information from the sources
2. If information is not in the sources, clearly state "This information is not available in the provided sources"
3. Use exact numbers, dates, and facts from sources when available
4. Organize your answer clearly with proper structure and complete sentences
5. Always end your response with a proper conclusion sentence
6. Cite sources where relevant: [Source 1], [Source 2], etc.

SOURCES:
{context}

QUESTION: {question}

COMPREHENSIVE ANSWER (based on sources, ensure complete response):"""

    
    CSV_DATA_PROMPT = """You are analyzing structured data from a CSV file. Be precise with data interpretation.

CSV ANALYSIS RULES:
1. Only reference data that actually exists in the CSV
2. Be precise with column names and values
3. For calculations, show your work using the actual data
4. If aggregations are needed, specify which rows/columns you're using
5. Distinguish between individual values and calculated summaries
6. Always cite sources: [Source 1], [Source 2], etc.

SOURCES:
{context}

QUESTION: {question}

DATA-DRIVEN ANSWER (show specific rows/columns referenced):"""

    @classmethod
    def get_prompt_for_content_type(cls, content_type: str, context: str, question: str) -> str:
        """Simplified prompt selection"""
        if content_type == 'csv':
            template = cls.CSV_DATA_PROMPT
        else:
            template = cls.STRICT_GROUNDING_PROMPT
        
        return template.format(context=context, question=question)
    
    @classmethod
    def detect_content_type(cls, context: str, question: str) -> str:
        """Detect the content type based on context and question"""
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Healthcare indicators
        healthcare_terms = ['patient', 'diagnosis', 'treatment', 'medical', 'disease', 'symptom', 'medication', 'clinical']
        if any(term in context_lower or term in question_lower for term in healthcare_terms):
            return 'healthcare'
        
        # Financial indicators
        financial_terms = ['revenue', 'profit', 'cost', 'budget', 'investment', 'financial', 'money', '$', '%']
        if any(term in context_lower or term in question_lower for term in financial_terms):
            return 'financial'
        
        # CSV/Data indicators
        if '|' in context and context.count('|') > 10:  # Table-like structure
            return 'csv'
        
        return 'default'

__all__ = ['AntiHallucinationPrompts']