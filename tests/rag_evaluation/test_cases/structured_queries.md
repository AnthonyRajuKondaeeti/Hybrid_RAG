# Test Cases for Structured Data (Employee Database CSV)

## Simple Queries (Direct Data Retrieval)

### Query 1: Employee Count
**Question**: "How many employees are in the dataset?"
**Expected Answer**: "There are 30 employees in the dataset."
**Key Facts**: 30 total employees
**Answer Type**: Count aggregation
**Complexity**: Simple

### Query 2: Specific Employee Information
**Question**: "What is Sarah Johnson's position and salary?"
**Expected Answer**: "Sarah Johnson is a Marketing Manager with a salary of $78,000."
**Key Facts**: Marketing Manager position, $78,000 salary
**Answer Type**: Individual record lookup
**Complexity**: Simple

### Query 3: Department Lookup
**Question**: "Which department does Michael Chen work in?"
**Expected Answer**: "Michael Chen works in the Engineering department."
**Key Facts**: Engineering department
**Answer Type**: Field lookup
**Complexity**: Simple

### Query 4: Highest Salary
**Question**: "Who has the highest salary in the company?"
**Expected Answer**: "Andrew Perez has the highest salary at $115,000, working as a Cloud Architect in Engineering."
**Key Facts**: Andrew Perez, $115,000, Cloud Architect
**Answer Type**: Maximum value query
**Complexity**: Simple

### Query 5: Location Count
**Question**: "How many employees work in San Francisco?"
**Expected Answer**: "There are 9 employees working in San Francisco."
**Key Facts**: 9 employees in San Francisco
**Answer Type**: Filtered count
**Complexity**: Simple

## Complex Queries (Multi-step Data Analysis)

### Query 6: Department Salary Analysis
**Question**: "Calculate the average salary by department and identify which department has the highest average compensation."
**Expected Answer**: "Engineering has the highest average salary at $88,111 (9 employees), followed by Sales at $72,600 (5 employees), Finance at $71,800 (5 employees), Marketing at $65,600 (5 employees), and HR at $59,750 (4 employees). Engineering's higher average reflects senior technical positions like Cloud Architect ($115,000) and Lead Developer ($110,000)."
**Key Facts**: Department averages, employee counts, salary ranges
**Answer Type**: Aggregation and analysis
**Complexity**: Complex

### Query 7: Performance and Experience Correlation
**Question**: "Analyze the relationship between years of experience and performance ratings, identifying any patterns or outliers."
**Expected Answer**: "There's a positive correlation between experience and performance ratings. Employees with 8+ years average 4.5 rating (Michael Chen 4.7, Andrew Perez 4.7, Nicole King 4.6), while 1-3 years experience average 3.8 rating. Notable outliers include Megan Nelson (6 years, 4.8 rating - highest overall) and Brandon Wright (1 year, 3.5 rating - lowest overall). Senior employees consistently perform above 4.0 ratings."
**Key Facts**: Experience-performance correlation, outlier identification, rating trends
**Answer Type**: Statistical analysis
**Complexity**: Complex

### Query 8: Geographic and Salary Distribution
**Question**: "Compare salary distributions across different locations and explain any geographic compensation patterns."
**Expected Answer**: "San Francisco has the highest average salary at $89,778 (9 employees) due to Engineering concentration and cost of living. Austin follows at $73,200 (4 employees), New York at $69,800 (6 employees), Chicago at $64,600 (5 employees), and Los Angeles at $62,000 (3 employees). The pattern reflects tech industry geographic premiums, with San Francisco commanding 44% higher salaries than Los Angeles."
**Key Facts**: Location-based salary analysis, cost of living factors, percentage differences
**Answer Type**: Geographic analysis
**Complexity**: Complex

### Query 9: Hiring Trends and Workforce Analysis
**Question**: "Analyze hiring patterns by year and department, identifying growth trends and strategic workforce changes."
**Expected Answer**: "Hiring peaked in 2020-2021 with 16 employees (53% of workforce), showing company growth. Engineering dominated hiring with 11 employees (37% of total workforce), indicating tech-focused expansion. 2019 had 7 hires, 2022 had 4 hires, and 2018 had 3 hires. Recent hiring (2021-2022) shows diversification into Sales and Marketing, suggesting business scaling beyond pure technical growth."
**Key Facts**: Yearly hiring patterns, department focus, strategic shifts
**Answer Type**: Trend analysis
**Complexity**: Complex

### Query 10: Compensation Equity and Optimization Analysis
**Question**: "Evaluate compensation equity across demographics and positions, and recommend optimization strategies for talent retention."
**Expected Answer**: "Performance-based analysis reveals high performers (4.5+ rating) earn $91,857 average vs $62,636 for lower performers, indicating merit-based compensation. Engineering commands premium salaries ($88,111 avg) reflecting market demands. Retention risks include top performers like Megan Nelson (4.8 rating, $98,000) and Michael Chen (4.7 rating, $110,000). Recommendations: implement performance bonuses for 4.5+ ratings, address potential underpayment in Marketing/HR, and create senior career paths to retain high performers."
**Key Facts**: Performance-compensation correlation, retention risks, strategic recommendations
**Answer Type**: Strategic workforce analysis
**Complexity**: Complex

## Evaluation Criteria

### Simple Query Success Metrics:
- **Data Accuracy**: Exact figures from CSV (95%+ target)
- **Calculation Correctness**: Proper aggregations and counts
- **Field Precision**: Correct column references
- **Format Consistency**: Appropriate data presentation

### Complex Query Success Metrics:
- **Analytical Rigor**: Multi-variable analysis (80%+ target)
- **Statistical Insight**: Meaningful pattern identification
- **Business Intelligence**: Actionable recommendations
- **Data Interpretation**: Context-aware conclusions

## CSV Data Reference
**Total Records**: 30 employees
**Departments**: Engineering (9), Marketing (3), Sales (5), HR (4), Finance (5)
**Locations**: San Francisco (9), New York (6), Chicago (5), Austin (4), Los Angeles (3)
**Salary Range**: $48,000 - $115,000
**Performance Range**: 3.5 - 4.8
**Experience Range**: 1 - 12 years