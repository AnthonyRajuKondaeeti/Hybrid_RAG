#!/usr/bin/env python3
"""
Test script to verify the generalized RAG system works for different types of documents.
Tests that the system no longer focuses on college/academic specific content.
"""

import os
import sys
import logging

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_service import HybridRAGService
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def test_generalized_rag():
    """Test the generalized RAG system with various query types"""
    try:
        # Initialize RAG service
        logger.info("Initializing RAG service...")
        config = Config()
        rag_service = HybridRAGService(config)
        
        # Use the most recent document (INTER.pdf)
        file_id = "44685a6c5417c74d3906762042aee98f"
        
        # Test various types of general questions
        test_questions = [
            # General summary questions
            "What is this document about?",
            "Provide a summary of the content",
            "What are the main points?",
            
            # Generic information extraction
            "What names are mentioned?",
            "What dates are mentioned?",
            "What contact information is available?",
            
            # General document analysis
            "What type of document is this?",
            "What key information does this contain?",
            "What are the important details?"
        ]
        
        logger.info(f"Testing generalized RAG for file_id: {file_id}")
        logger.info("=" * 60)
        
        success_count = 0
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- Test {i}: {question} ---")
            
            try:
                result = rag_service.answer_question(
                    question=question,
                    file_id=file_id,
                    session_id="test_general_rag"
                )
                
                success = result.get('success', False)
                answer = result.get('answer', 'No answer')
                
                logger.info(f"Success: {success}")
                
                if success:
                    success_count += 1
                    logger.info(f"Answer: {answer}")
                    
                    # Check if answer seems generic and helpful
                    if len(answer.strip()) > 10 and not answer.strip() == ".":
                        logger.info("✅ Answer appears meaningful and generic")
                    else:
                        logger.warning("⚠️ Answer seems too brief or minimal")
                        
                    # Check for domain-specific bias (should not heavily focus on academic terms)
                    academic_terms = ['college', 'university', 'grade', 'student', 'academic']
                    answer_lower = answer.lower()
                    academic_count = sum(1 for term in academic_terms if term in answer_lower)
                    
                    if academic_count <= 1:  # Allow some mentions but not heavy focus
                        logger.info("✅ Answer appears domain-neutral")
                    else:
                        logger.warning(f"⚠️ Answer may be biased toward academic domain ({academic_count} academic terms)")
                        
                else:
                    logger.warning(f"❌ Query failed: {answer}")
                    
            except Exception as e:
                logger.error(f"Error during test {i}: {str(e)}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"SUMMARY: {success_count}/{len(test_questions)} queries succeeded")
        
        if success_count >= len(test_questions) * 0.7:  # 70% success rate
            logger.info("✅ Generalized RAG system appears to be working well")
        else:
            logger.warning("⚠️ Generalized RAG system may need further improvements")
        
        # Test the fallback system specifically
        logger.info("\n--- Testing Fallback System ---")
        logger.info("Testing with a question that should trigger fallback logic...")
        
        fallback_result = rag_service.answer_question(
            question="Give me a detailed summary with all key information",
            file_id=file_id,
            session_id="test_fallback"
        )
        
        if fallback_result.get('fallback_used', False):
            logger.info("✅ Fallback system was triggered successfully")
            logger.info(f"Fallback answer: {fallback_result.get('answer', '')[:200]}...")
        else:
            logger.info("ℹ️ LLM response worked normally (no fallback needed)")
            
    except Exception as e:
        logger.error(f"Error initializing RAG service: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting generalized RAG system test...")
    success = test_generalized_rag()
    if success:
        logger.info("Generalized RAG system test completed")
    else:
        logger.error("Generalized RAG system test failed")