import pandas as pd
import os
from itertools import combinations
from analysis_engine import SurveyAnalyzerSingleDriver, SurveyAnalyzerDriversComparison

# --- SCALABLE CONFIGURATION ---

# 1. Define your generic, reusable recommendation templates based on themes.
RECOMMENDATION_MAP = {
    'default': "Overall performance is stable. Continue monitoring key metrics and supporting leadership communication efforts.",
    'leadership_trust': (
        "**Increase Leadership Visibility and Trust:** Confidence in leadership is a concern. "
        "Launch initiatives to build trust through transparency, such as hosting more authentic, unscripted Q&A sessions with the Executive Team "
        "and ensuring leaders are actively addressing employee feedback."
    ),
    'manager_support': (
        "**Strengthen Manager Support Systems:** Manager support appears to be a weak point. "
        "Implement a dedicated program to equip managers with the tools, talking points, and training necessary to lead their teams effectively through the transformation. "
        "This is a critical leverage point for success."
    ),
    'resource_allocation': (
        "**Review and Communicate Resource Allocation:** Concerns about the availability of time and resources are evident. "
        "Leadership should conduct a transparent review of project roadmaps and resource allocation to ensure teams are not overburdened. "
        "Communicate the outcomes of this review to alleviate concerns."
    ),
    'vision_clarity': (
        "**Enhance Vision Clarity and Purpose:** The understanding of the vision or its purpose is low. "
        "Shift communication from *what* is changing to *why* it's essential for the company's future. "
        "Use multiple channels and storytelling to connect the transformation to the company's core mission."
    ),
    'communication_effectiveness': (
        "**Overhaul a Key Communication Channel:** A primary communication channel (like the intranet or email) is underperforming. "
        "Charter a project to gather user feedback and revamp this channel to ensure it meets employee needs and serves as a reliable source of information."
    )
}

# 2. Define each survey driver you want to analyze.
ANALYSIS_CONFIG = [
    {
        'driver_name': "Vision and Direction",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["adb|vision_lv1", "adb|vision_agree_lv1", "adb|conf_lv1_ldr", "enb|awareness", "adb|understand_purpose"],
        'metric_themes': {
            "adb|vision_lv1": "vision_clarity",
            "adb|vision_agree_lv1": "vision_clarity",
            "adb|conf_lv1_ldr": "leadership_trust",
            "enb|awareness": "vision_clarity",
            "adb|understand_purpose": "vision_clarity"
        }
    },
    {
        'driver_name': "Communication",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["adb|info_managers", "adb|info_written", "ada|info_intranet"],
        'metric_themes': {
            "adb|info_managers": "manager_support",
            "adb|info_written": "communication_effectiveness",
            "ada|info_intranet": "communication_effectiveness"
        }
    },
    {
        'driver_name': "Business Leadership",
        'data_file': "data/TGPS Learning Activity Cycle.xlsx",
        'metric_columns': ["enb|ldr_support_system", "enb|ldr_time_resources", "rsb|quick_remedial", "sfb|current_change_mgmt", "enb|conf_lv2_ldr"],
        'metric_themes': {
            "enb|ldr_support_system": "manager_support",
            "enb|ldr_time_resources": "resource_allocation",
            "rsb|quick_remedial": "manager_support",
            "sfb|current_change_mgmt": "manager_support",
            "enb|conf_lv2_ldr": "leadership_trust"
        }
    }
]

# 3. Set to True to generate a combined insights report
GENERATE_COMPARISON = True
OUTPUT_DIR = "output"

# --- EXECUTION ---
def main():
    """Main function to run the survey analysis pipeline."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    all_key_findings = []
    
    # --- Individual Report Generation ---
    for config in ANALYSIS_CONFIG:
        driver_name = config['driver_name']
        print(f"Analyzing driver: {driver_name}...")
        try:
            df_qcode = pd.read_excel(config['data_file'], header=None)
            df = pd.read_excel(config['data_file'], skiprows=1)
        except FileNotFoundError:
            print(f"Error: Data file not found at {config['data_file']}. Skipping.")
            all_key_findings.append({'driver_name': driver_name, 'error': True}) # Add placeholder
            continue
            
        analyzer_single = SurveyAnalyzerSingleDriver(
            df=df,
            df_qcode=df_qcode,
            driver_name=driver_name,
            metric_columns=config['metric_columns'],
            metric_themes=config['metric_themes'],
            recommendation_map=RECOMMENDATION_MAP
        )
        report_content = analyzer_single.generate_report()
        all_key_findings.append(analyzer_single.get_key_findings())
        
        file_name = driver_name.lower().replace(" ", "_") + ".md"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Successfully generated report: {output_path}")

    # --- Pairwise Comparison Report Generation ---
    valid_findings = [f for f in all_key_findings if 'error' not in f]
    if len(valid_findings) > 1 and GENERATE_COMPARISON:
        print("\nGenerating pairwise comparison reports...")
        for combo in combinations(valid_findings, 2):
            findings_a, findings_b = combo

            analyser_comparison = SurveyAnalyzerDriversComparison(findings_a, findings_b)
            
            comparison_content = analyser_comparison.generate_pairwise_comparison_report()
            
            name_a = findings_a['driver_name'].lower().replace(" ", "_")
            name_b = findings_b['driver_name'].lower().replace(" ", "_")
            file_name = f"comparison_{name_a}_vs_{name_b}.md"
            output_path = os.path.join(OUTPUT_DIR, file_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(comparison_content)
            print(f"Successfully generated comparison report: {output_path}")

if __name__ == "__main__":
    main()