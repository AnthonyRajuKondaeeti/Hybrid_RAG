#!/bin/bash
# Complete RAG Performance Test Script

echo "🚀 Starting Comprehensive RAG Performance Testing..."

cd rag_testing/testing_scripts

echo "📊 Running full test suite..."
python test_runner.py --full-suite

echo "⚡ Running performance benchmarks..."
python test_runner.py --benchmark --iterations 5

echo "📈 Analyzing latest results..."
cd ../
python complete_improvement_test.py

echo "✅ Complete! Check the test_results/ folder for detailed reports."