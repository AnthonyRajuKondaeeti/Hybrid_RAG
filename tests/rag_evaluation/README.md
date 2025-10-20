# RAG System Evaluation Framework

A comprehensive testing and evaluation framework for the Xplorease RAG system, designed to provide quantitative analysis and validation of retrieval-augmented generation performance.

## 📁 Directory Structure

```
rag_evaluation/
├── documents/           # Test documents by category
│   ├── technical/       # Software docs, APIs, manuals
│   ├── business/        # Reports, policies, presentations  
│   ├── academic/        # Research papers, case studies
│   └── structured/      # CSV, JSON, tabular data
├── test_cases/         # Query sets with expected answers
├── evaluators/         # Scoring algorithms and metrics
├── reports/           # Generated evaluation reports
└── automation/        # Test automation and CI/CD
```

## 🎯 Evaluation Metrics

### Accuracy Metrics (40% weight)
- Factual correctness validation
- Source attribution accuracy  
- Information completeness

### Relevance Metrics (25% weight)
- Query-answer alignment
- Context appropriateness
- Information specificity

### Quality Metrics (20% weight)
- Response coherence
- Language clarity
- Structural organization

### Technical Metrics (15% weight)
- Response time performance
- Confidence scores
- Source coverage

## 🚀 Quick Start

```bash
# Run complete evaluation suite
python automation/run_evaluation.py

# Generate executive report
python automation/generate_report.py --format executive

# Run specific document type tests
python automation/run_evaluation.py --category technical
```

## 📊 Reports Generated

- **Executive Summary**: High-level performance overview
- **Detailed Analysis**: Per-category breakdowns
- **Technical Metrics**: Performance and reliability data
- **Comparison Reports**: Before/after analysis

## 🎖️ Success Criteria

- **Simple Queries**: >85% accuracy target
- **Complex Queries**: >75% accuracy target  
- **Response Time**: <2 seconds average
- **Relevance**: >90% information retrieval
- **Hallucination Rate**: <5% target