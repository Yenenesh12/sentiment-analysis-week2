#!/usr/bin/env python3
"""
Task 4: Insights and Recommendations Main Script
Generates insights, visualizations, and final report.
"""

import pandas as pd
import logging
from datetime import datetime
import os
from src.database.database_connection import DatabaseConnection
from src.analysis.insights import BankingInsightsAnalyzer
from src.visualization.plot_generator import BankingVisualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_final_report(report_data: dict, output_path: str = './reports/final_report.md'):
    """Generate final report in Markdown format"""
    
    with open(output_path, 'w') as f:
        f.write(f"""# Banking Apps Sentiment Analysis - Final Report

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Total Reviews Analyzed:** {sum(c['total_reviews'] for c in report_data['bank_comparison'].values())}

## Executive Summary

{report_data['executive_summary']}

## 1. Bank Comparison

### Performance Metrics
""")
        
        # Bank comparison table
        f.write("| Bank | Total Reviews | Avg Rating | Avg Sentiment | Positive % |\n")
        f.write("|------|--------------|------------|---------------|------------|\n")
        
        for bank, stats in report_data['bank_comparison'].items():
            f.write(f"| {bank} | {stats['total_reviews']} | {stats['avg_rating']:.2f} | "
                   f"{stats['avg_sentiment']:.3f} | {stats['positive_pct']:.1f}% |\n")
        
        f.write("""
## 2. Satisfaction Drivers and Pain Points

### Commercial Bank of Ethiopia (CBE)
**Drivers:**
""")
        
        for driver in report_data['drivers_pain_points'].get('Commercial Bank of Ethiopia', {}).get('drivers', []):
            f.write(f"- **{driver['category'].replace('_', ' ').title()}**: {driver['count']} mentions\n")
            if driver.get('sample_reviews'):
                f.write(f"  *Sample review:* \"{driver['sample_reviews'][0][:100]}...\"\n")
        
        f.write("\n**Pain Points:**\n")
        for pain in report_data['drivers_pain_points'].get('Commercial Bank of Ethiopia', {}).get('pain_points', []):
            f.write(f"- **{pain['category'].replace('_', ' ').title()}**: {pain['count']} mentions\n")
            if pain.get('sample_reviews'):
                f.write(f"  *Sample review:* \"{pain['sample_reviews'][0][:100]}...\"\n")
        
        f.write("""
### Bank of Abyssinia (BOA)
**Drivers:**
""")
        
        for driver in report_data['drivers_pain_points'].get('Bank of Abyssinia', {}).get('drivers', []):
            f.write(f"- **{driver['category'].replace('_', ' ').title()}**: {driver['count']} mentions\n")
            if driver.get('sample_reviews'):
                f.write(f"  *Sample review:* \"{driver['sample_reviews'][0][:100]}...\"\n")
        
        f.write("\n**Pain Points:**\n")
        for pain in report_data['drivers_pain_points'].get('Bank of Abyssinia', {}).get('pain_points', []):
            f.write(f"- **{pain['category'].replace('_', ' ').title()}**: {pain['count']} mentions\n")
            if pain.get('sample_reviews'):
                f.write(f"  *Sample review:* \"{pain['sample_reviews'][0][:100]}...\"\n")
        
        f.write("""
### Dashen Bank
**Drivers:**
""")
        
        for driver in report_data['drivers_pain_points'].get('Dashen Bank', {}).get('drivers', []):
            f.write(f"- **{driver['category'].replace('_', ' ').title()}**: {driver['count']} mentions\n")
            if driver.get('sample_reviews'):
                f.write(f"  *Sample review:* \"{driver['sample_reviews'][0][:100]}...\"\n")
        
        f.write("\n**Pain Points:**\n")
        for pain in report_data['drivers_pain_points'].get('Dashen Bank', {}).get('pain_points', []):
            f.write(f"- **{pain['category'].replace('_', ' ').title()}**: {pain['count']} mentions\n")
            if pain.get('sample_reviews'):
                f.write(f"  *Sample review:* \"{pain['sample_reviews'][0][:100]}...\"\n")
        
        f.write("""
## 3. Recommendations

### CBE Recommendations:
""")
        
        for rec in report_data['recommendations'].get('Commercial Bank of Ethiopia', []):
            f.write(f"- **[{rec['priority'].upper()}]** {rec['recommendation']} ({rec['category']})\n")
        
        f.write("""
### BOA Recommendations:
""")
        
        for rec in report_data['recommendations'].get('Bank of Abyssinia', []):
            f.write(f"- **[{rec['priority'].upper()}]** {rec['recommendation']} ({rec['category']})\n")
        
        f.write("""
### Dashen Bank Recommendations:
""")
        
        for rec in report_data['recommendations'].get('Dashen Bank', []):
            f.write(f"- **[{rec['priority'].upper()}]** {rec['recommendation']} ({rec['category']})\n")
        
        f.write("""
## 4. Ethical Considerations

""")
        
        for bias in report_data['ethics_analysis'].get('review_biases', []):
            f.write(f"- {bias}\n")
        
        for issue in report_data['ethics_analysis'].get('sampling_issues', []):
            f.write(f"- {issue}\n")
        
        for consideration in report_data['ethics_analysis'].get('ethical_considerations', []):
            f.write(f"- {consideration}\n")
        
        f.write("""
## 5. Key Findings

""")
        
        for finding in report_data.get('key_findings', []):
            f.write(f"- {finding}\n")
        
        f.write("""
## 6. Conclusion

This analysis provides actionable insights for improving mobile banking applications in Ethiopia. 
The recommendations focus on addressing key pain points while enhancing existing strengths. 
Regular monitoring of user feedback and continuous improvement based on these insights will 
help banks improve customer satisfaction and maintain competitive advantage.

---

**Note:** All visualizations are saved in the `reports/plots/` directory.
""")
    
    logger.info(f"Final report saved to {output_path}")
    return output_path

def main():
    """Main function for Task 4"""
    
    print("="*60)
    print("TASK 4: INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    # Initialize database connection
    db = DatabaseConnection()
    
    try:
        # Connect to database
        if not db.connect():
            logger.error("Failed to connect to database")
            return
        
        print("\n📊 Loading and analyzing data...")
        
        # Initialize analyzer
        analyzer = BankingInsightsAnalyzer(db)
        
        # Generate comprehensive report
        report_data = analyzer.generate_full_report()
        
        if not report_data:
            logger.error("Failed to generate report data")
            return
        
        print("✅ Analysis completed")
        
        # Load data for visualizations
        df = analyzer.load_review_data()
        trends_df = analyzer.analyze_sentiment_trends(df)
        drivers_data = report_data['drivers_pain_points']
        
        print("\n🎨 Generating visualizations...")
        
        # Generate visualizations
        viz = BankingVisualizations()
        viz.save_all_visualizations(df, trends_df, drivers_data)
        
        print("✅ Visualizations saved")
        
        # Generate final report
        print("\n📝 Generating final report...")
        report_path = generate_final_report(report_data)
        
        print("\n" + "="*60)
        print("TASK 4 COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\n📋 OUTPUT SUMMARY:")
        print(f"   📊 Analysis Report: {report_path}")
        print(f"   🎨 Visualizations: ./reports/plots/")
        print(f"   📈 Insights Generated: {len(report_data.get('key_findings', []))}")
        print(f"   💡 Recommendations: {sum(len(recs) for recs in report_data.get('recommendations', {}).values())}")
        
        # Print key insights
        print("\n🔑 KEY INSIGHTS:")
        for finding in report_data.get('key_findings', []):
            print(f"   • {finding}")
        
        print("\n🎯 RECOMMENDATIONS PRIORITY:")
        for bank, recs in report_data.get('recommendations', {}).items():
            high_priority = sum(1 for r in recs if r['priority'] == 'high')
            print(f"   {bank}: {high_priority} high-priority recommendations")
        
    except Exception as e:
        logger.error(f"Task 4 execution failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        db.close()

if __name__ == "__main__":
    main()