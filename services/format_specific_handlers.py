"""
Format-Specific Handlers for RAG System Improvements

This module provides specialized handling for different document formats
that showed poor performance in the evaluation:
- CSV: 0.374 (worst performer)
- HTML: 0.403 (second worst)
- Healthcare content: Very low scores across formats
"""

import re
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

class CSVDataHandler:
    """Specialized handler for CSV/structured data questions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_csv_prompt(self, question: str, csv_content: str) -> str:
        """Create specialized prompt for CSV data questions"""
        
        # Parse CSV content to understand structure
        structure_info = self._analyze_csv_structure(csv_content)
        
        prompt = f"""You are analyzing structured data from a CSV file. Be precise with data interpretation.

CSV DATA STRUCTURE:
- Columns: {structure_info['columns']}
- Total Rows: {structure_info['row_count']}
- Data Types: {structure_info['data_types']}

CSV ANALYSIS RULES:
1. Only reference data that actually exists in the CSV
2. Be precise with column names and values
3. For calculations, show your work using the actual data
4. If aggregations are needed, specify which rows/columns you're using
5. Distinguish between individual values and calculated summaries

CSV DATA:
{csv_content}

QUESTION: {question}

DATA-DRIVEN ANSWER (show specific rows/columns referenced):"""
        
        return prompt
    
    def _analyze_csv_structure(self, csv_content: str) -> Dict[str, Any]:
        """Analyze CSV structure for better prompting"""
        try:
            lines = csv_content.strip().split('\n')
            if not lines:
                return {"columns": [], "row_count": 0, "data_types": {}}
            
            # Parse header
            header = lines[0].split(',')
            columns = [col.strip().strip('"') for col in header]
            
            # Analyze data types from first few rows
            data_types = {}
            if len(lines) > 1:
                for i, col in enumerate(columns):
                    sample_values = []
                    for line in lines[1:min(6, len(lines))]:  # Sample first 5 data rows
                        values = line.split(',')
                        if i < len(values):
                            sample_values.append(values[i].strip().strip('"'))
                    
                    data_types[col] = self._infer_column_type(sample_values)
            
            return {
                "columns": columns,
                "row_count": len(lines) - 1,  # Subtract header
                "data_types": data_types
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing CSV structure: {e}")
            return {"columns": [], "row_count": 0, "data_types": {}}
    
    def _infer_column_type(self, values: List[str]) -> str:
        """Infer column data type from sample values"""
        if not values:
            return "unknown"
        
        # Check if all values are numeric
        numeric_count = 0
        date_count = 0
        
        for value in values:
            if not value:
                continue
            
            # Check numeric
            try:
                float(value.replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except ValueError:
                pass
            
            # Check date patterns
            if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value) or \
               re.match(r'\d{4}-\d{2}-\d{2}', value):
                date_count += 1
        
        total = len([v for v in values if v])
        if total == 0:
            return "empty"
        
        if numeric_count / total > 0.8:
            return "numeric"
        elif date_count / total > 0.8:
            return "date"
        else:
            return "text"


class HTMLContentHandler:
    """Specialized handler for HTML content questions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_html_prompt(self, question: str, html_content: str) -> str:
        """Create specialized prompt for HTML content questions"""
        
        # Clean and structure HTML content
        cleaned_content = self._clean_html_content(html_content)
        structure_info = self._analyze_html_structure(html_content)
        
        prompt = f"""You are analyzing web content from an HTML document. Focus on the actual text content.

HTML CONTENT STRUCTURE:
- Main sections: {structure_info['sections']}
- Contains tables: {structure_info['has_tables']}
- Contains lists: {structure_info['has_lists']}
- Key headings: {structure_info['headings'][:5]}  # First 5 headings

WEB CONTENT ANALYSIS RULES:
1. Focus on the actual text content, ignore HTML formatting
2. If referencing tables, be specific about which table and which data
3. For lists, reference specific list items
4. Distinguish between headings and body content
5. If information spans multiple sections, cite the relevant sections

CLEANED CONTENT:
{cleaned_content}

QUESTION: {question}

WEB CONTENT ANSWER (cite specific sections/elements):"""
        
        return prompt
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content for better processing"""
        try:
            # Remove HTML tags but preserve structure
            import re
            
            # Replace common HTML elements with text markers
            html_content = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'[HEADING] \1', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<td[^>]*>(.*?)</td>', r'[\1]', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<th[^>]*>(.*?)</th>', r'[HEADER: \1]', html_content, flags=re.IGNORECASE)
            
            # Remove all remaining HTML tags
            clean_text = re.sub(r'<[^>]+>', ' ', html_content)
            
            # Clean up whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            return clean_text
            
        except Exception as e:
            self.logger.warning(f"Error cleaning HTML content: {e}")
            return html_content
    
    def _analyze_html_structure(self, html_content: str) -> Dict[str, Any]:
        """Analyze HTML structure for better prompting"""
        try:
            import re
            
            # Find headings
            headings = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', html_content, flags=re.IGNORECASE)
            headings = [re.sub(r'<[^>]+>', '', h).strip() for h in headings]
            
            # Check for structural elements
            has_tables = bool(re.search(r'<table', html_content, flags=re.IGNORECASE))
            has_lists = bool(re.search(r'<[uo]l', html_content, flags=re.IGNORECASE))
            
            # Identify main sections
            sections = []
            if headings:
                sections = headings[:10]  # First 10 headings as sections
            
            return {
                "headings": headings,
                "sections": sections,
                "has_tables": has_tables,
                "has_lists": has_lists
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing HTML structure: {e}")
            return {"headings": [], "sections": [], "has_tables": False, "has_lists": False}


class HealthcareContentHandler:
    """Specialized handler for healthcare content - needs highest precision"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.medical_terms = {
            'hba1c': 'hemoglobin A1C',
            'bp': 'blood pressure', 
            'systolic': 'systolic blood pressure',
            'diastolic': 'diastolic blood pressure',
            'mmhg': 'millimeters of mercury',
            'mg/dl': 'milligrams per deciliter',
            'diabetes': 'diabetes mellitus',
            'type 1': 'type 1 diabetes',
            'type 2': 'type 2 diabetes'
        }
    
    def create_healthcare_prompt(self, question: str, content: str) -> str:
        """Create ultra-precise prompt for healthcare content"""
        
        # Identify medical entities in content
        medical_entities = self._identify_medical_entities(content)
        
        prompt = f"""You are providing healthcare information based on medical documentation. EXTREME PRECISION IS REQUIRED.

MEDICAL ENTITIES IDENTIFIED:
{json.dumps(medical_entities, indent=2)}

CRITICAL HEALTHCARE RULES:
1. NEVER provide medical advice or recommendations
2. Only state exact values, ranges, and guidelines as written in the source
3. Always include units of measurement (mg/dL, mmHg, %, etc.)
4. Distinguish between target values, normal ranges, and clinical thresholds
5. If specific populations are mentioned (adults, elderly, pregnant), be explicit
6. State the source of guidelines if mentioned (ADA, WHO, etc.)
7. Never extrapolate or interpret medical data
8. If information is incomplete, explicitly state what's missing

MEDICAL SOURCE CONTENT:
{content}

MEDICAL QUESTION: {question}

PRECISE MEDICAL ANSWER (include exact values and units):"""
        
        return prompt
    
    def _identify_medical_entities(self, content: str) -> Dict[str, List[str]]:
        """Identify medical entities in content for better handling"""
        content_lower = content.lower()
        
        entities = {
            "measurements": [],
            "conditions": [],
            "medications": [],
            "values_with_units": [],
            "guidelines": []
        }
        
        # Find values with medical units
        import re
        
        # Find numerical values with units
        unit_patterns = [
            r'\d+\.?\d*\s*mg/dl',
            r'\d+\.?\d*\s*mmhg', 
            r'\d+\.?\d*\s*%',
            r'\d+\.?\d*\s*mg',
            r'\d+\.?\d*\s*units?',
            r'\d+\.?\d*\s*times?\s+(?:daily|weekly|monthly)'
        ]
        
        for pattern in unit_patterns:
            matches = re.findall(pattern, content_lower)
            entities["values_with_units"].extend(matches)
        
        # Find medical conditions
        condition_keywords = ['diabetes', 'hypertension', 'hypoglycemia', 'hyperglycemia']
        for keyword in condition_keywords:
            if keyword in content_lower:
                entities["conditions"].append(keyword)
        
        # Find measurement types
        measurement_keywords = ['hba1c', 'blood pressure', 'glucose', 'cholesterol']
        for keyword in measurement_keywords:
            if keyword in content_lower:
                entities["measurements"].append(keyword)
        
        # Find guideline sources
        guideline_sources = ['ada', 'who', 'american diabetes association', 'world health']
        for source in guideline_sources:
            if source in content_lower:
                entities["guidelines"].append(source)
        
        return entities


class FormatSpecificRAGHandler:
    """Main handler that routes to format-specific processors"""
    
    def __init__(self):
        self.csv_handler = CSVDataHandler()
        self.html_handler = HTMLContentHandler()
        self.healthcare_handler = HealthcareContentHandler()
        self.logger = logging.getLogger(__name__)
    
    def create_format_specific_prompt(self, question: str, content: str, 
                                    file_format: str, content_type: str = "general") -> str:
        """Create prompts optimized for specific formats and content types"""
        
        try:
            # Healthcare content gets special treatment regardless of format
            if content_type == "healthcare" or self._is_healthcare_content(question, content):
                return self.healthcare_handler.create_healthcare_prompt(question, content)
            
            # Format-specific handling
            if file_format.lower() in ['.csv', 'csv']:
                return self.csv_handler.create_csv_prompt(question, content)
            
            elif file_format.lower() in ['.html', '.htm', 'html']:
                return self.html_handler.create_html_prompt(question, content)
            
            else:
                # Default improved prompt for other formats
                return self._create_general_improved_prompt(question, content, file_format)
                
        except Exception as e:
            self.logger.error(f"Error creating format-specific prompt: {e}")
            return self._create_fallback_prompt(question, content)
    
    def _is_healthcare_content(self, question: str, content: str) -> bool:
        """Detect if content is healthcare-related"""
        healthcare_indicators = [
            'hba1c', 'diabetes', 'blood pressure', 'medication', 'medical',
            'patient', 'treatment', 'clinical', 'diagnosis', 'mmhg', 'mg/dl',
            'systolic', 'diastolic', 'glucose', 'insulin'
        ]
        
        combined_text = (question + " " + content).lower()
        return any(indicator in combined_text for indicator in healthcare_indicators)
    
    def _create_general_improved_prompt(self, question: str, content: str, file_format: str) -> str:
        """Create improved general prompt with format awareness"""
        
        format_guidance = {
            '.pdf': "This is PDF content which may include technical specifications, structured data, or formatted text.",
            '.docx': "This is document content which may include headings, bullet points, and formatted sections.",
            '.xlsx': "This is spreadsheet content which may include tabular data, calculations, and structured information.",
            '.txt': "This is plain text content. Focus on the actual text and any implied structure.",
            '.md': "This is markdown content which may include headings, lists, and formatted sections.",
            '.png': "This content was extracted from an image using OCR and may contain formatting artifacts."
        }
        
        guidance = format_guidance.get(file_format, "Analyze this content carefully.")
        
        return f"""You are analyzing content from a {file_format} file. {guidance}

ANALYSIS RULES:
1. Only use information explicitly stated in the content
2. Be precise with any numbers, dates, or technical terms
3. If referencing specific sections, identify them clearly
4. Maintain the context and structure of the original content
5. If information is unclear due to formatting, state this explicitly

CONTENT TO ANALYZE:
{content}

QUESTION: {question}

PRECISE ANSWER (reference specific parts of the content):"""
    
    def _create_fallback_prompt(self, question: str, content: str) -> str:
        """Fallback prompt when format-specific handling fails"""
        return f"""Answer the following question using only the provided content. Be precise and cite specific information.

CONTENT:
{content}

QUESTION: {question}

ANSWER:"""


# Integration function for existing RAG service
def integrate_format_handlers_with_rag(rag_service):
    """
    Integration function to add format-specific handling to existing RAG service
    
    Usage:
        rag_service = YourExistingRAGService(config)
        integrate_format_handlers_with_rag(rag_service)
    """
    
    # Add format handler to the service
    rag_service.format_handler = FormatSpecificRAGHandler()
    
    # Store original method
    if hasattr(rag_service, '_generate_answer_with_llm'):
        rag_service._original_generate_answer = rag_service._generate_answer_with_llm
    
    def enhanced_generate_answer(self, question, context_chunks, file_formats=None):
        """Enhanced answer generation with format-specific handling"""
        
        try:
            # Determine dominant format
            dominant_format = self._get_dominant_format(file_formats) if file_formats else "general"
            
            # Combine context
            combined_content = '\n\n'.join(context_chunks[:5])  # Limit to 5 chunks
            
            # Detect content type
            content_type = self._detect_content_type_enhanced(question, combined_content)
            
            # Create format-specific prompt
            enhanced_prompt = self.format_handler.create_format_specific_prompt(
                question=question,
                content=combined_content,
                file_format=dominant_format,
                content_type=content_type
            )
            
            # Generate answer using existing LLM
            return self._original_generate_answer(enhanced_prompt)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced answer generation: {e}")
            # Fallback to original method
            return self._original_generate_answer(f"Context: {combined_content}\n\nQuestion: {question}")
    
    def _get_dominant_format(self, file_formats):
        """Get the most common format from results"""
        if not file_formats:
            return "general"
        
        format_counts = {}
        for fmt in file_formats:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        return max(format_counts, key=format_counts.get) if format_counts else "general"
    
    def _detect_content_type_enhanced(self, question, content):
        """Enhanced content type detection"""
        combined = (question + " " + content).lower()
        
        if any(term in combined for term in ['hba1c', 'diabetes', 'blood pressure', 'medical']):
            return "healthcare"
        elif any(term in combined for term in ['revenue', 'profit', 'financial', 'quarter']):
            return "financial"  
        elif any(term in combined for term in ['api', 'system', 'technical', 'architecture']):
            return "technical"
        else:
            return "general"
    
    # Bind methods to the service
    rag_service.enhanced_generate_answer = enhanced_generate_answer.__get__(rag_service)
    rag_service._get_dominant_format = _get_dominant_format.__get__(rag_service)
    rag_service._detect_content_type_enhanced = _detect_content_type_enhanced.__get__(rag_service)
    
    return rag_service


# Export classes
__all__ = [
    'FormatSpecificRAGHandler', 
    'CSVDataHandler', 
    'HTMLContentHandler', 
    'HealthcareContentHandler',
    'integrate_format_handlers_with_rag'
]