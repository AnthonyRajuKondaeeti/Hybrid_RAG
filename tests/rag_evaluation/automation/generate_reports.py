"""
RAG Evaluation Report Generator

This module creates comprehensive reports with charts and statistics 
for managerial review and decision-making.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import statistics

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    
    # Set style for professional reports
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
except ImportError:
    print("⚠ Matplotlib/Seaborn not available - text reports only")
    PLOTTING_AVAILABLE = False


class RAGReportGenerator:
    """Generate comprehensive reports for RAG evaluation results"""
    
    def __init__(self, results_dir: str = None):
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = Path(__file__).parent.parent / "reports"
        
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_latest_results(self) -> Dict[str, Any]:
        """Load the most recent evaluation results"""
        # Look for both naming patterns
        result_files = list(self.results_dir.glob("evaluation_results_*.json")) + \
                      list(self.results_dir.glob("rag_evaluation_results_*.json"))
        
        if not result_files:
            raise FileNotFoundError("No evaluation results found. Run evaluation first.")
            
        # Get the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading results from: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the results"""
        stats = {
            'overview': {},
            'by_category': {},
            'by_complexity': {},
            'performance_metrics': {},
            'quality_analysis': {}
        }
        
        # Flatten results if they're organized by category
        all_results = []
        if isinstance(results, dict) and any(isinstance(v, list) for v in results.values()):
            # Results are organized by category
            for category, category_results in results.items():
                if isinstance(category_results, list):
                    for result in category_results:
                        result['category'] = category
                        all_results.append(result)
        elif 'results' in results:
            # Results have a 'results' key
            all_results = results['results']
        else:
            # Assume results is a list
            all_results = results if isinstance(results, list) else []
        
        # Filter successful results
        successful_results = [r for r in all_results if r.get('overall_score', 0) > 0]
        
        # Overall statistics
        if successful_results:
            all_scores = [r['overall_score'] for r in successful_results]
            all_times = [r.get('response_time', 0) for r in successful_results]
            
            stats['overview'] = {
                'total_queries': len(all_results),
                'successful_queries': len(successful_results),
                'success_rate': len(successful_results) / len(all_results) * 100 if all_results else 0,
                'overall_score_mean': statistics.mean(all_scores),
                'overall_score_median': statistics.median(all_scores),
                'overall_score_std': statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                'avg_response_time': statistics.mean(all_times) if all_times else 0,
                'response_time_std': statistics.stdev(all_times) if len(all_times) > 1 else 0,
                'excellent_rate': len([s for s in all_scores if s >= 0.8]) / len(all_scores) * 100,
                'good_rate': len([s for s in all_scores if s >= 0.6]) / len(all_scores) * 100
            }
        else:
            stats['overview'] = {
                'total_queries': len(all_results),
                'successful_queries': 0,
                'success_rate': 0,
                'overall_score_mean': 0,
                'overall_score_median': 0,
                'overall_score_std': 0,
                'avg_response_time': 0,
                'response_time_std': 0,
                'excellent_rate': 0,
                'good_rate': 0
            }
        
        # Category analysis
        categories = ['technical', 'business', 'academic', 'structured']
        for category in categories:
            cat_results = [r for r in successful_results if r.get('category') == category]
            if cat_results:
                cat_scores = [r['overall_score'] for r in cat_results]
                cat_times = [r.get('response_time', 0) for r in cat_results]
                
                stats['by_category'][category] = {
                    'count': len(cat_results),
                    'mean_score': statistics.mean(cat_scores),
                    'median_score': statistics.median(cat_scores),
                    'std_score': statistics.stdev(cat_scores) if len(cat_scores) > 1 else 0,
                    'avg_time': statistics.mean(cat_times) if cat_times else 0,
                    'excellent_rate': len([s for s in cat_scores if s >= 0.8]) / len(cat_scores) * 100
                }
        
        # Performance metrics breakdown
        if successful_results:
            metric_names = ['accuracy_score', 'relevance_score', 'quality_score', 'technical_score']
            for metric in metric_names:
                metric_scores = [r.get(metric, 0) for r in successful_results]
                if metric_scores and any(s > 0 for s in metric_scores):
                    clean_name = metric.replace('_score', '')
                    stats['performance_metrics'][clean_name] = {
                        'mean': statistics.mean(metric_scores),
                        'median': statistics.median(metric_scores),
                        'std': statistics.stdev(metric_scores) if len(metric_scores) > 1 else 0,
                        'min': min(metric_scores),
                        'max': max(metric_scores)
                    }
        
        return stats
    
    def generate_html_report(self, results: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Generate a comprehensive HTML report"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .category-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .category-table th, .category-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .category-table th {{
            background-color: #3498db;
            color: white;
        }}
        .category-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .score-excellent {{ color: #27ae60; font-weight: bold; }}
        .score-good {{ color: #f39c12; font-weight: bold; }}
        .score-needs-improvement {{ color: #e74c3c; font-weight: bold; }}
        .insights-box {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .recommendation {{
            background: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 RAG System Evaluation Report</h1>
        <p style="text-align: center; color: #7f8c8d;">
            Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
        </p>
        
        <h2>📊 Executive Summary</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{stats['overview']['total_queries']}</div>
                <div class="metric-label">Total Test Queries</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['overview']['overall_score_mean']:.1%}</div>
                <div class="metric-label">Average Performance</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['overview']['success_rate']:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['overview']['avg_response_time']:.2f}s</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['overview']['excellent_rate']:.1f}%</div>
                <div class="metric-label">Excellent Scores (≥80%)</div>
            </div>
        </div>
        
        <h2>📈 Performance by Category</h2>
        <table class="category-table">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Test Count</th>
                    <th>Avg Score</th>
                    <th>Excellence Rate</th>
                    <th>Avg Response Time</th>
                    <th>Performance Rating</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add category rows
        for category, data in stats['by_category'].items():
            score = data['mean_score']
            score_class = ('score-excellent' if score >= 0.8 else 
                          'score-good' if score >= 0.6 else 
                          'score-needs-improvement')
            
            rating = ('🌟 Excellent' if score >= 0.8 else 
                     '✅ Good' if score >= 0.6 else 
                     '⚠️ Needs Improvement')
            
            excellence_rate = data.get('excellence_rate', 0)
            avg_time = data.get('avg_time', 0)
            
            html_content += f"""
                <tr>
                    <td style="text-transform: capitalize;">{category}</td>
                    <td>{data['count']}</td>
                    <td class="{score_class}">{score:.1%}</td>
                    <td>{excellence_rate:.1f}%</td>
                    <td>{avg_time:.2f}s</td>
                    <td>{rating}</td>
                </tr>
            """
        
        html_content += f"""
            </tbody>
        </table>
        
        <h2>🎯 Performance Metrics Breakdown</h2>
        <div class="summary-grid">
        """
        
        # Add metric breakdown
        for metric, data in stats['performance_metrics'].items():
            html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{data['mean']:.1%}</div>
                <div class="metric-label">{metric.title()} Score</div>
            </div>
            """
        
        html_content += f"""
        </div>
        
        <h2>💡 Key Insights & Recommendations</h2>
        <div class="insights-box">
            <h3>Performance Analysis:</h3>
            <ul>
        """
        
        # Generate insights based on data
        best_category = max(stats['by_category'].items(), key=lambda x: x[1]['mean_score'])
        worst_category = min(stats['by_category'].items(), key=lambda x: x[1]['mean_score'])
        
        html_content += f"""
                <li><strong>Best Performing Category:</strong> {best_category[0].title()} ({best_category[1]['mean_score']:.1%} average score)</li>
                <li><strong>Area for Improvement:</strong> {worst_category[0].title()} ({worst_category[1]['mean_score']:.1%} average score)</li>
                <li><strong>Overall System Performance:</strong> {'Above expectations' if stats['overview']['overall_score_mean'] >= 0.7 else 'Meets basic requirements' if stats['overview']['overall_score_mean'] >= 0.5 else 'Requires optimization'}</li>
                <li><strong>Response Time:</strong> {'Excellent' if stats['overview']['avg_response_time'] <= 0.5 else 'Acceptable' if stats['overview']['avg_response_time'] <= 1.0 else 'May need optimization'} ({stats['overview']['avg_response_time']:.2f}s average)</li>
            </ul>
        </div>
        
        <div class="recommendation">
            <h3>🎯 Strategic Recommendations:</h3>
            <ol>
        """
        
        # Generate recommendations based on performance
        if stats['overview']['overall_score_mean'] < 0.6:
            html_content += "<li><strong>Priority:</strong> Implement comprehensive model fine-tuning and knowledge base optimization</li>"
        
        if worst_category[1]['mean_score'] < best_category[1]['mean_score'] - 0.15:
            html_content += f"<li><strong>Focus Area:</strong> Enhance {worst_category[0]} domain expertise and training data</li>"
        
        if stats['overview']['avg_response_time'] > 0.5:
            html_content += "<li><strong>Performance:</strong> Optimize query processing pipeline for faster response times</li>"
        
        if stats['overview']['excellent_rate'] < 30:
            html_content += "<li><strong>Quality:</strong> Implement advanced evaluation metrics and feedback loops</li>"
        
        html_content += f"""
                <li><strong>Monitoring:</strong> Establish regular evaluation cycles and performance tracking dashboards</li>
            </ol>
        </div>
        
        <h2>📋 Technical Summary</h2>
        <div class="insights-box">
            <h3>Test Configuration:</h3>
            <ul>
                <li><strong>Total Test Cases:</strong> {stats['overview']['total_queries']} across 4 categories</li>
                <li><strong>Document Types:</strong> Technical docs, Business reports, Academic papers, Structured data</li>
                <li><strong>Query Complexity:</strong> Mix of simple and complex queries</li>
                <li><strong>Evaluation Metrics:</strong> Accuracy (40%), Relevance (25%), Quality (20%), Technical (15%)</li>
                <li><strong>Success Criteria:</strong> ≥60% for acceptable, ≥80% for excellent performance</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by RAG Evaluation Framework v1.0</p>
            <p>For technical details, see the full evaluation results JSON file</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def save_charts(self, results: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
        """Generate and save visualization charts"""
        if not PLOTTING_AVAILABLE:
            print("⚠ Plotting libraries not available - skipping charts")
            return []
        
        chart_files = []
        
        # Get all results in a flat list
        all_results = []
        if isinstance(results, dict) and any(isinstance(v, list) for v in results.values()):
            for category, category_results in results.items():
                if isinstance(category_results, list):
                    for result in category_results:
                        result['category'] = category
                        all_results.append(result)
        
        if not all_results:
            print("⚠ No results data for charts")
            return []
        
        successful_results = [r for r in all_results if r.get('overall_score', 0) > 0]
        
        # 1. Overall Performance Distribution
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Score distribution
        plt.subplot(2, 2, 1)
        scores = [r['overall_score'] for r in successful_results]
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Score Distribution')
        plt.xlabel('Overall Score')
        plt.ylabel('Frequency')
        if scores:
            plt.axvline(statistics.mean(scores), color='red', linestyle='--', label='Mean')
            plt.legend()
        
        # Subplot 2: Performance by Category
        plt.subplot(2, 2, 2)
        categories = list(stats['by_category'].keys())
        cat_scores = [stats['by_category'][cat]['mean_score'] for cat in categories]
        bars = plt.bar(categories, cat_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Performance by Category')
        plt.ylabel('Average Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, cat_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.1%}', ha='center', va='bottom')
        
        # Subplot 3: Response Time Distribution
        plt.subplot(2, 2, 3)
        times = [r.get('response_time', 0) for r in successful_results]
        times = [t for t in times if t > 0]  # Filter out zero times
        if times:
            plt.hist(times, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.axvline(statistics.mean(times), color='red', linestyle='--', label='Mean')
            plt.legend()
        
        # Subplot 4: Metrics Breakdown
        plt.subplot(2, 2, 4)
        if stats['performance_metrics']:
            metric_names = list(stats['performance_metrics'].keys())
            metric_scores = [stats['performance_metrics'][metric]['mean'] for metric in metric_names]
            plt.pie(metric_scores, labels=metric_names, autopct='%1.1f%%', startangle=90)
            plt.title('Performance Metrics Breakdown')
        
        plt.tight_layout()
        chart_file = self.results_dir / f"performance_overview_{self.timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(str(chart_file))
        
        return chart_files
    
    def generate_full_report(self) -> Dict[str, str]:
        """Generate a complete report with all components"""
        print("📊 Generating comprehensive RAG evaluation report...")
        
        # Load results
        results = self.load_latest_results()
        
        # Count total results for debug
        total_results = 0
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, list):
                    total_results += len(value)
        
        print(f"✓ Loaded results for {total_results} queries")
        
        # Calculate statistics
        stats = self.calculate_statistics(results)
        print("✓ Calculated performance statistics")
        
        # Generate HTML report
        html_content = self.generate_html_report(results, stats)
        html_file = self.results_dir / f"rag_evaluation_report_{self.timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Generated HTML report: {html_file.name}")
        
        # Generate charts
        chart_files = self.save_charts(results, stats)
        print(f"✓ Generated {len(chart_files)} visualization charts")
        
        # Save detailed statistics as JSON
        stats_file = self.results_dir / f"detailed_statistics_{self.timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"✓ Saved detailed statistics: {stats_file.name}")
        
        # Create executive summary text file
        summary_file = self.results_dir / f"executive_summary_{self.timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_summary(stats))
        print(f"✓ Created executive summary: {summary_file.name}")
        
        print(f"\n🎉 Report generation complete!")
        print(f"📁 All files saved to: {self.results_dir}")
        
        return {
            'html_report': str(html_file),
            'statistics': str(stats_file),
            'summary': str(summary_file),
            'charts': chart_files
        }
    
    def _generate_text_summary(self, stats: Dict[str, Any]) -> str:
        """Generate a concise text summary for executives"""
        
        summary = f"""
RAG SYSTEM EVALUATION - EXECUTIVE SUMMARY
========================================
Generated: {datetime.now().strftime("%B %d, %Y")}

KEY PERFORMANCE INDICATORS:
---------------------------
• Overall Performance Score: {stats['overview']['overall_score_mean']:.1%}
• System Success Rate: {stats['overview']['success_rate']:.1f}%
• Average Response Time: {stats['overview']['avg_response_time']:.2f} seconds
• Excellence Rate (≥80%): {stats['overview']['excellent_rate']:.1f}%

CATEGORY PERFORMANCE:
--------------------
"""
        
        for category, data in stats['by_category'].items():
            status = "🌟 EXCELLENT" if data['mean_score'] >= 0.8 else "✅ GOOD" if data['mean_score'] >= 0.6 else "⚠️ NEEDS ATTENTION"
            summary += f"• {category.upper()}: {data['mean_score']:.1%} - {status}\n"
        
        # Determine overall assessment
        overall_score = stats['overview']['overall_score_mean']
        if overall_score >= 0.8:
            assessment = "EXCELLENT - System exceeds performance expectations"
        elif overall_score >= 0.6:
            assessment = "GOOD - System meets business requirements"
        else:
            assessment = "REQUIRES IMPROVEMENT - Optimization needed"
        
        summary += f"""
OVERALL ASSESSMENT: {assessment}

BUSINESS IMPACT:
---------------
• The RAG system demonstrates {'strong' if overall_score >= 0.7 else 'adequate' if overall_score >= 0.5 else 'limited'} capability for production deployment
• Response times are {'excellent' if stats['overview']['avg_response_time'] <= 0.5 else 'acceptable' if stats['overview']['avg_response_time'] <= 1.0 else 'concerning'} for user experience
• Quality consistency is {'high' if stats['overview']['overall_score_std'] <= 0.15 else 'moderate' if stats['overview']['overall_score_std'] <= 0.25 else 'variable'}

RECOMMENDATIONS:
---------------
• {'Continue with current configuration' if overall_score >= 0.8 else 'Implement targeted improvements' if overall_score >= 0.6 else 'Require significant optimization before production'}
• Monitor performance trends and establish regular evaluation cycles
• Focus optimization efforts on {'lowest performing categories' if max(stats['by_category'].values(), key=lambda x: x['mean_score'])['mean_score'] - min(stats['by_category'].values(), key=lambda x: x['mean_score'])['mean_score'] > 0.15 else 'maintaining current performance levels'}
"""
        
        return summary


def main():
    """Main function to generate reports"""
    try:
        generator = RAGReportGenerator()
        report_files = generator.generate_full_report()
        
        print("\n" + "="*60)
        print("📋 REPORT GENERATION SUMMARY")
        print("="*60)
        print(f"HTML Report: {Path(report_files['html_report']).name}")
        print(f"Statistics: {Path(report_files['statistics']).name}")
        print(f"Summary: {Path(report_files['summary']).name}")
        
        if report_files['charts']:
            print("Charts:")
            for chart in report_files['charts']:
                print(f"  - {Path(chart).name}")
        
        print(f"\n💼 Manager-ready files available in: {Path(__file__).parent.parent / 'reports'}")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()