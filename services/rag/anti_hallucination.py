"""
Anti-Hallucination Components

This module contains specialized prompts and utilities designed to minimize
hallucinations and improve answer quality in the RAG system.
"""

class AntiHallucinationPrompts:
    """Prompts designed to minimize hallucinations and improve answer quality"""
    
    STRICT_GROUNDING_PROMPT = """You are a precise fact-checker. Answer ONLY using information from the provided sources.

CRITICAL RULES:
1. If information is not in the sources, say "This information is not available in the provided sources"
2. Never add your own knowledge or make assumptions
3. Quote exact numbers, dates, and facts from sources
4. If unsure, explicitly state your uncertainty
5. Always cite which source you're using: [Source 1], [Source 2], etc.

SOURCES:
{context}

QUESTION: {question}

ANSWER (stick strictly to sources):"""

    HEALTHCARE_PROMPT = """You are answering a healthcare question. Be extremely precise and never guess.

MEDICAL SAFETY RULES:
1. Only provide information explicitly stated in the medical sources
2. Never extrapolate or suggest medical advice
3. If specific medical values are not in sources, state this clearly
4. Use exact medical terminology from sources
5. Always cite sources: [Source 1], [Source 2], etc.

SOURCES:
{context}

QUESTION: {question}

MEDICAL ANSWER (source-based only):"""

    FINANCIAL_PROMPT = """You are analyzing financial data. Precision is critical.

FINANCIAL ACCURACY RULES:
1. Only use exact numbers and percentages from the sources
2. Never estimate or approximate financial figures
3. Clearly state the time period and context for any numbers
4. If financial data is incomplete, explicitly state what's missing
5. Always cite sources: [Source 1], [Source 2], etc.

SOURCES:
{context}

QUESTION: {question}

FINANCIAL ANSWER (exact figures only):"""

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
        """Get the appropriate prompt template based on content type"""
        prompt_mapping = {
            'healthcare': cls.HEALTHCARE_PROMPT,
            'financial': cls.FINANCIAL_PROMPT,
            'csv': cls.CSV_DATA_PROMPT,
            'default': cls.STRICT_GROUNDING_PROMPT
        }
        
        template = prompt_mapping.get(content_type.lower(), cls.STRICT_GROUNDING_PROMPT)
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