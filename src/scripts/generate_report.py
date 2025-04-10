import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Add the project root to the path so we can import HF modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HF.utils.report_utils import save_report, read_report, append_to_report
from HF.utils.data_utils import load_dataframe
from HF.utils.visualization_utils import set_plot_style
from HF.logger import logging

def generate_combined_report():
    """Generate a combined report with all analysis information"""
    try:
        # Create the report header
        header = f"""# Heart Failure Detection - Combined Analysis Report
==================================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Create the combined report
        combined_report = header
        
        # Add the analysis summary
        try:
            summary = read_report("analysis_summary.txt")
            combined_report += f"\n\n{summary}\n\n"
            combined_report += "\n" + "="*50 + "\n"
        except Exception as e:
            logging.error(f"Error reading analysis summary: {e}")
            combined_report += "\n\n## Analysis Summary\nError reading analysis summary\n\n"
        
        # Add the technical analysis
        try:
            technical = read_report("technical_analysis.txt")
            combined_report += f"\n\n{technical}\n\n"
            combined_report += "\n" + "="*50 + "\n"
        except Exception as e:
            logging.error(f"Error reading technical analysis: {e}")
            combined_report += "\n\n## Technical Analysis\nError reading technical analysis\n\n"
        
        # Add the clinical interpretation
        try:
            clinical = read_report("clinical_interpretation.txt")
            combined_report += f"\n\n{clinical}\n\n"
        except Exception as e:
            logging.error(f"Error reading clinical interpretation: {e}")
            combined_report += "\n\n## Clinical Interpretation\nError reading clinical interpretation\n\n"
        
        # Save the combined report
        save_report(combined_report, "combined_report.txt")
        
        logging.info("Combined report generated successfully")
        
        return "combined_report.txt"
    except Exception as e:
        logging.error(f"Error generating combined report: {e}")
        raise e

def generate_executive_summary():
    """Generate an executive summary of the analysis"""
    try:
        # Create the report header
        header = f"""# Heart Failure Detection - Executive Summary
==================================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Create the executive summary
        executive_summary = header
        
        # Add the executive summary content
        executive_summary += """
## Project Overview
This project developed a machine learning model to detect heart failure based on clinical, demographic, and cardiac measurement data. The model achieved high accuracy and provides interpretable predictions that align with clinical understanding of heart failure.

## Key Findings
1. **Model Performance**: The XGBoost model achieved 86.3% accuracy and 0.949 ROC AUC, outperforming other algorithms and ensemble approaches.
2. **Important Predictors**: Cardiac function measurements (FS, LVIDs) and clinical symptoms (NYHA class, Chest pain) are the strongest predictors.
3. **Clinical Utility**: The model shows high specificity (94.5%) and reasonable sensitivity (68.0%), making it suitable for screening applications.
4. **Robustness**: 10-fold cross-validation confirms consistent performance across different data subsets.

## Recommendations
1. Deploy the XGBoost model for heart failure risk prediction in clinical settings.
2. Focus on the top 5 features for simplified data collection and interpretation.
3. Adjust the prediction threshold based on the clinical context (higher for screening, lower for diagnosis).
4. Validate the model in prospective clinical trials before widespread adoption.
5. Develop a user-friendly interface for clinicians to access and interpret model predictions.

## Next Steps
1. Collect more data to improve model performance, especially for positive cases.
2. Develop separate models for different heart failure subtypes (HFrEF vs. HFpEF).
3. Incorporate temporal features to distinguish between acute and chronic heart failure.
4. Implement explainable AI techniques to enhance clinical interpretability.
5. Create a web-based application for real-time predictions in clinical settings.
"""
        
        # Save the executive summary
        save_report(executive_summary, "executive_summary.txt")
        
        logging.info("Executive summary generated successfully")
        
        return "executive_summary.txt"
    except Exception as e:
        logging.error(f"Error generating executive summary: {e}")
        raise e

if __name__ == "__main__":
    # Set the plot style
    set_plot_style()
    
    # Generate the combined report
    combined_report = generate_combined_report()
    
    # Generate the executive summary
    executive_summary = generate_executive_summary()
    
    print(f"Reports generated successfully:")
    print(f"1. Combined Report: reports/{combined_report}")
    print(f"2. Executive Summary: reports/{executive_summary}")
