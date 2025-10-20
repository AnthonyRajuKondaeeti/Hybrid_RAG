# Test Cases for Technical Documentation (API Documentation)

## Simple Queries (Direct Fact Retrieval)

### Query 1: Basic Installation Information
**Question**: "What are the minimum system requirements for XploreaseAPI?"
**Expected Answer**: "The minimum system requirements are Python 3.8 or higher, 4GB RAM (8GB recommended), 2GB free storage space, and Windows 10+, macOS 10.15+, or Ubuntu 18.04+."
**Key Facts**: Python 3.8+, 4GB RAM minimum, 2GB storage, OS requirements
**Answer Type**: Factual retrieval
**Complexity**: Simple

### Query 2: API Rate Limiting
**Question**: "What is the rate limit for the authentication endpoint?"
**Expected Answer**: "The authentication endpoint (/api/v2/auth/login) has a rate limit of 5 requests per minute."
**Key Facts**: 5 requests per minute, authentication endpoint
**Answer Type**: Specific technical detail
**Complexity**: Simple

### Query 3: File Upload Constraints
**Question**: "What is the maximum file size for document uploads?"
**Expected Answer**: "The maximum file size for document uploads is 50MB, with support for PDF, DOCX, TXT, CSV, and JSON formats."
**Key Facts**: 50MB limit, supported formats
**Answer Type**: Technical specification
**Complexity**: Simple

### Query 4: Default Configuration
**Question**: "What is the default database port in the configuration?"
**Expected Answer**: "The default database port is 5432."
**Key Facts**: Port 5432
**Answer Type**: Configuration detail
**Complexity**: Simple

### Query 5: Support Contact
**Question**: "What is the emergency support phone number?"
**Expected Answer**: "The emergency hotline is +1-800-XPLOREASE."
**Key Facts**: Emergency contact number
**Answer Type**: Contact information
**Complexity**: Simple

## Complex Queries (Multi-step Reasoning & Analysis)

### Query 6: Performance Optimization Analysis
**Question**: "If search queries are taking more than 2 seconds, what troubleshooting steps should be taken and what performance targets should be expected after optimization?"
**Expected Answer**: "For slow query performance (>2 seconds), the issue is likely database indexing problems or large result sets. The solution is to rebuild search indexes using 'xplorease-admin reindex'. After optimization, expect average response times of 150ms for search queries with a target uptime of 99.9%."
**Key Facts**: >2 second symptom, indexing cause, reindex solution, 150ms target
**Answer Type**: Troubleshooting analysis
**Complexity**: Complex

### Query 7: Security Implementation Strategy
**Question**: "What security measures must be implemented for a production deployment and what are the maintenance requirements?"
**Expected Answer**: "Production deployments require SSL/TLS encryption (mandatory), API key rotation every 90 days, session timeout of 30 minutes, and passwords with minimum 12 characters including alphanumeric and special characters. Regular maintenance includes key rotation and security monitoring."
**Key Facts**: SSL/TLS required, 90-day rotation, 30-min timeout, password policy
**Answer Type**: Security strategy
**Complexity**: Complex

### Query 8: Capacity Planning Analysis
**Question**: "Based on the performance metrics, how many concurrent users can the system handle and what would be the infrastructure requirements?"
**Expected Answer**: "Based on 1000 requests per minute throughput and 150ms average response time, the system can handle approximately 166 concurrent users (1000 req/min ÷ 60 sec = 16.67 req/sec ÷ 0.15 sec = ~111 users, with overhead ~166). Infrastructure requires minimum 8GB RAM (preferably more for large document processing), database with 20 connection pool size, and proper indexing."
**Key Facts**: 1000 req/min, 150ms response, capacity calculation
**Answer Type**: Performance analysis
**Complexity**: Complex

### Query 9: Integration Implementation Guide
**Question**: "What would be the complete integration process for a new application including authentication, file upload, and search functionality?"
**Expected Answer**: "Complete integration requires: 1) Authentication via POST /api/v2/auth/login with username/password to obtain JWT token, 2) Document upload via POST /api/v2/documents/upload (max 50MB, formats: PDF/DOCX/TXT/CSV/JSON), 3) Search implementation via GET /api/v2/documents/search with query parameter. Consider rate limits (5 auth requests/minute), processing time (30-120 seconds), and response times (200-500ms for search)."
**Key Facts**: Three-step process, endpoint details, constraints
**Answer Type**: Integration guide
**Complexity**: Complex

### Query 10: Cost-Benefit Analysis for Deployment
**Question**: "What are the operational costs and benefits of deploying XploreaseAPI in a high-availability production environment?"
**Expected Answer**: "High-availability deployment requires infrastructure supporting 99.9% uptime target, processing 50 documents per minute with 1000 requests per minute throughput. Benefits include 150ms average search response time and enterprise-grade security. Costs include 24/7 monitoring, regular API key rotation every 90 days, and sufficient hardware (8GB+ RAM, indexed database). The system supports up to 3 concurrent uploads and provides fuzzy matching with semantic search capabilities."
**Key Facts**: 99.9% uptime, processing rates, response times, resource requirements
**Answer Type**: Cost-benefit analysis
**Complexity**: Complex

## Evaluation Criteria

### Simple Query Success Metrics:
- **Accuracy**: Exact fact retrieval (90%+ target)
- **Completeness**: All key facts included
- **Relevance**: Direct answer to question
- **Speed**: Response time <1 second

### Complex Query Success Metrics:
- **Accuracy**: Multi-fact synthesis (75%+ target)
- **Reasoning**: Logical step-by-step analysis
- **Completeness**: Comprehensive solution coverage
- **Practicality**: Actionable recommendations