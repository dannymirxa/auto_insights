import pandas as pd
import numpy as np
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from insights import data
from insights import findings
from insights import hotspots
from insights import correlations

# --- The SurveyAnalyzer Class (Unchanged from previous version) ---
class SurveyAnalyzerSingleDriver:
    """
    Analyzes a single survey transformation driver from a pandas DataFrame,
    generating a rich, narrative-driven insights report based on a provided configuration.
    """
    def __init__(self, df: pd.DataFrame, df_qcode: pd.DataFrame, driver_name: str, metric_columns: list, metric_themes: dict, recommendation_map: dict):
        self.driver_name = driver_name
        self.metric_columns = metric_columns
        self.metric_themes = metric_themes
        self.recommendation_map = recommendation_map
        self.demographic_cols = ['Job Level', 'Length of Service', 'Function', 'Sub-Function']
        self.df = data.preprocess_df(df.copy(), metric_columns=self.metric_columns, demographic_cols=self.demographic_cols)
        self.df_questions_qcode = data.preprocess_question_qcode(df_qcode.copy())
        self.key_findings = {}
       
    def generate_report(self) -> str:
        """
        highest_metric = self.key_findings['highest_metric']
        lowest_metric = self.key_findings['lowest_metric']
        Build an insights report. This method is resilient to:
        - missing metric columns
        - metric columns that are all non-numeric / NaN
        It attempts to produce as much useful output as possible and documents
        which metrics were unavailable so the caller can diagnose data issues.
        """
        report_parts = [f"# Insights Report: {self.driver_name}\n"]
 
        # Check metric presence and numeric availability
        missing_columns = [c for c in self.metric_columns if c not in self.df.columns]
        numeric_available = []
        non_numeric_or_empty = []
        for c in self.metric_columns:
            if c not in self.df.columns:
                continue
            # consider a column numeric-available if at least one non-NaN numeric value exists
            col_vals = pd.to_numeric(self.df[c], errors='coerce')
            if col_vals.dropna().shape[0] > 0:
                numeric_available.append(c)
            else:
                non_numeric_or_empty.append(c)
 
        if missing_columns or non_numeric_or_empty:
            report_parts.append("## Data Availability\n")
            if missing_columns:
                report_parts.append(
                    "- The following expected metric columns are missing from the dataset: "
                    + ", ".join(missing_columns) + "."
                )
            if non_numeric_or_empty:
                report_parts.append(
                    "- The following metric columns are present but contain no numeric values: "
                    + ", ".join(non_numeric_or_empty) + "."
                )
            report_parts.append(
                "\n\nBecause some metrics are missing or non-numeric, the report will use whatever valid metrics are available."
            )
 
        # If no numeric metrics exist at all, emit a clear diagnostic report
        if not numeric_available:
            report_parts.append("\n\n## Executive Summary\n")
            report_parts.append(
                "Could not generate quantitative insights because none of the configured metric columns "
                "contain numeric data. Please check the input file and column names used in ANALYSIS_CONFIG."
            )
            report_parts.append("\n\n---\n\n## Key Observations\n")
            report_parts.append("- No numeric metrics were available to compute scores.")
            return "\n".join(report_parts)
 
        # Compute overall scores using only numeric-available metrics
        overall_scores = self.df[numeric_available].mean().dropna().to_dict()
        sorted_scores = sorted(overall_scores.items(), key=lambda item: item[1], reverse=True)
 
        # Populate key findings (only for available metrics)
        self.key_findings['driver_name'] = self.driver_name
        self.key_findings['metric_themes'] = self.metric_themes
        self.key_findings['lowest_metric'] = {'name': sorted_scores[-1][0], 'score': sorted_scores[-1][1]}
        self.key_findings['highest_metric'] = {'name': sorted_scores[0][0], 'score': sorted_scores[0][1]}
        self.key_findings['average_score'] = float(np.mean(list(overall_scores.values())))
        spots_by_function_and_joblevel = hotspots.hotspots_and_bright_spots_by_function_and_joblevel(self.df, self.metric_columns, num_spots=3)
        spots_by_drivers = hotspots.hotspots_and_bright_spots_by_function_and_driver(self.df, self.metric_columns, num_spots=3)
        spots_by_function = hotspots.hotspots_and_bright_spots_by_function(self.df, self.metric_columns, num_spots=3)
        spots_by_joblevel = hotspots.hotspots_and_bright_spots_by_joblevel(self.df, self.metric_columns, num_spots=3)
        self.key_findings.update(spots_by_function_and_joblevel)
        self.key_findings.update(spots_by_drivers)
        self.key_findings.update(spots_by_function)
        self.key_findings.update(spots_by_joblevel)
 
        # Build a concise but useful report body
        report_parts.append(findings.executive_summary(
            self.driver_name,
            self.key_findings['average_score'],
            len(numeric_available)
        ))
        report_parts.append("\n\n---\n\n## Key Observations\n")
        # report_parts.append(f"- Highest scoring metric: **{self.key_findings['highest_metric']['name']}** "
        #                     f"({self.key_findings['highest_metric']['score']:.2f}).")
        # report_parts.append(f"- Lowest scoring metric: **{self.key_findings['lowest_metric']['name']}** "
        #                     f"({self.key_findings['lowest_metric']['score']:.2f}).")
        key_obs_lines = findings.key_observations(self.key_findings['highest_metric'], self.key_findings['lowest_metric'])
        report_parts.extend(key_obs_lines)
 
        # Hotspots / Bright spots by Function & Job Level
        hotspots_by_function_and_joblevel = self.key_findings.get('hotspots_by_function_and_joblevel', [])
        bright_spots = self.key_findings.get('bright_spots_by_function_and_joblevel', [])
        if bright_spots:
            report_parts.append("\n\n## Bright Spots (by Function and Job Level)\n")
            for spot in bright_spots:
                report_parts.append(f"- {spot['Function']} ({spot['Job Level']}): {spot['average_score']}")
        if hotspots_by_function_and_joblevel:
            report_parts.append("\n\n## Hotspots (by Function and Job Level)\n")
            for spot in hotspots_by_function_and_joblevel:
                report_parts.append(f"- {spot['Function']} ({spot['Job Level']}): {spot['average_score']}")

        # Hotspots / Bright spots by Function
        hotspots_by_function = self.key_findings.get('hotspots_by_function', [])
        bright_spots_by_function = self.key_findings.get('bright_spots_by_function', [])
        if bright_spots_by_function:
            report_parts.append("\n\n## Bright Spots (by Function)\n")
            for spot in bright_spots_by_function:
                report_parts.append(f"- {spot['Function']}: {spot['average_score']}")
        if hotspots_by_function:
            report_parts.append("\n\n## Hotspots (by Function)\n")
            for spot in hotspots_by_function:
                report_parts.append(f"- {spot['Function']}: {spot['average_score']}")

        # Hotspots / Bright spots by Job Level
        hotspots_by_joblevel = self.key_findings.get('hotspots_by_joblevel', [])
        bright_spots_by_joblevel = self.key_findings.get('bright_spots_by_joblevel', [])
        if bright_spots_by_joblevel:
            report_parts.append("\n\n## Bright Spots (by Job Level)\n")
            for spot in bright_spots_by_joblevel:
                report_parts.append(f"- {spot['Job Level']}: {spot['average_score']}")
        if hotspots_by_joblevel:
            report_parts.append("\n\n## Hotspots (by Job Level)\n")
            for spot in hotspots_by_joblevel:
                report_parts.append(f"- {spot['Job Level']}: {spot['average_score']}")

        # Hotspots / Bright spots by Driver (per-metric Function-level results)
        hotspots_by_driver = self.key_findings.get('hotspots_by_driver', [])
        bright_spots_by_driver = self.key_findings.get('bright_spots_by_driver', [])
        if bright_spots_by_driver:
            report_parts.append("\n\n## Bright Spots (by Metric & Function)\n")
            for spot in bright_spots_by_driver:
                # spot has ['Function','Metric','average_score']
                func = spot.get('Function', 'Unknown')
                metric = spot.get('Metric', 'Unknown')
                avg = spot.get('average_score')
                report_parts.append(f"- {func} (metric: {metric}): {avg}")
        if hotspots_by_driver:
            report_parts.append("\n\n## Hotspots (by Metric & Function)\n")
            for spot in hotspots_by_driver:
                func = spot.get('Function', 'Unknown')
                metric = spot.get('Metric', 'Unknown')
                avg = spot.get('average_score')
                report_parts.append(f"- {func} (metric: {metric}): {avg}")
 
        # --- Correlations / Heatmap (use only numeric-available metrics) ---
        try:
            corr_matrix = correlations.correlations_matrix(self.df, self.metric_columns).loc[numeric_available, numeric_available]
            # if the correlation matrix has any non-NaN values, include it
            if not corr_matrix.isna().all().all():
                report_parts.append("\n\n## Metric Correlations\n")
                report_parts.append(
                    "The table below shows pairwise Pearson correlations between the available metrics. "
                    "Emojis indicate strength and direction: 🟩 strong positive, 🟢 moderate positive, 🟡 weak positive, ⚪ neutral/none, 🔴 negative.\n"
                )
                report_parts.append(correlations.format_heatmap(corr_matrix))
                # report_parts
        except Exception:
            # Do not fail the whole report if correlations cannot be computed
            report_parts.append("\n\n## Metric Correlations\n")
            report_parts.append("- Could not compute correlations for the available metrics.")
        
        # --- Correlations between other Q-codes (drivers) and this single driver ---
        try:
            # correlations_between_qcode_and_single_driver returns a DataFrame with columns:
            # ['questions', 'qcode', 'score'].
            # We pass numeric_available (only metrics that have numeric data).
            qcode_corrs = correlations.correlations_between_qcode_and_single_driver(self.df, self.df_questions_qcode, numeric_available)
            # Ensure qcode strings are safe for markdown tables by escaping literal pipes
            qcode_corrs['qcode'] = qcode_corrs['qcode'].astype(str).str.replace("|", "\|")
            if qcode_corrs is not None and not qcode_corrs.empty:
                report_parts.append("\n\n## Correlations: Other Q-codes vs. This Driver\n")
                report_parts.append(
                    "This table shows Pearson correlations between the per-respondent average score for the configured metrics "
                    "and other q-code (driver) columns in the dataset. Scores are rounded to two decimals and sorted highest-to-lowest."
                )
                # Render the full DataFrame as a markdown table including questions, qcode and score.
                report_parts.append("\n\n| Question | Q-code | Correlation |\n|----|----|----|")
                for _, row in qcode_corrs.iterrows():
                    question = row.get('question', '')
                    qcode = row.get('qcode', '')
                    score = row.get('score', '')
                    try:
                        report_parts.append(f"| {question} | {qcode} | {float(score):.2f} |")
                    except Exception:
                        report_parts.append(f"| {question} | {qcode} | {score} |")
        except Exception:
            # Don't fail the whole report if this additional correlation analysis fails.
            report_parts.append("\n\n## Correlations: Other Q-codes vs. This Driver\n")
            report_parts.append("- Could not compute correlations between q-codes and this driver.")
        try:
            recommendations = findings.recommendations_from_theme(self.key_findings['lowest_metric']['name'], self.metric_themes, self.recommendation_map)
            report_parts.append("\n\n## Recommendations\n")
            report_parts.append(recommendations)
        except Exception:
            # If for some reason recommendation generation fails, don't crash report creation.
            report_parts.append("\n\n## Recommendations\n")
            report_parts.append("- Could not generate recommendations due to incomplete theme mapping.")
 
        return "\n".join(report_parts)

    def get_key_findings(self) -> dict:
        if 'hotspots' not in self.key_findings:
            self.generate_report()
        return self.key_findings
    
class SurveyAnalyzerDriversComparison:
    def __init__(self, findings_A, findings_B):
        self.findings_A = findings_A
        self.findings_B = findings_B

    def generate_pairwise_comparison_report(self) -> str:
        """
        Generates a deep, narrative-driven comparison between two drivers.
        """
        name_A, name_B = self.findings_A['driver_name'], self.findings_B['driver_name']
        report_parts = [f"# Comparison Report: **{name_A}** vs. **{name_B}**\n"]

        # --- Executive Summary ---
        report_parts.append(
            "## Executive Summary\n\n"
            f"This report synthesizes the findings from the **{name_A}** and **{name_B}** surveys to uncover deeper, interconnected insights. "
            "The analysis reveals a codependent relationship between these two areas, where successes and failures in one driver directly impact the other. "
            "Understanding this dynamic is crucial for developing effective, systemic interventions rather than addressing symptoms in isolation."
        )

        # --- Helper to get themes ---
        def get_themes(findings):
            themes = findings.get('metric_themes', {})
            lowest_metric_name = findings['lowest_metric']['name']
            highest_metric_name = findings['highest_metric']['name']
            return {
                'lowest': themes.get(lowest_metric_name),
                'highest': themes.get(highest_metric_name)
            }
        
        themes_A, themes_B = get_themes(self.findings_A), get_themes(self.findings_B)

        # --- The Reinforcing Loop (Positive Story) ---
        report_parts.append(
            "\n\n---\n\n## The Reinforcing Loop: What's Working Well ✨\n\n"
            "By examining the strongest areas of each driver, we can identify a positive feedback loop that should be amplified.\n"
        )
        positive_story = (
            f"The primary strength in the **{name_A}** survey is **{themes_A['highest'].replace('_', ' ')}** "
            f"(driven by the metric '{self.findings_A['highest_metric']['name']}' with a score of {self.findings_A['highest_metric']['score']:.2f}). "
            f"This is strongly complemented by the success in **{themes_B['highest'].replace('_', ' ')}** from the **{name_B}** survey "
            f"(metric: '{self.findings_B['highest_metric']['name']}', score: {self.findings_B['highest_metric']['score']:.2f}).\n\n"
            "**Insight**: This suggests that these two areas are working in concert. For example, strong manager support may be effectively amplifying the clarity of the company vision. "
            "We should study the teams and leaders who exemplify this positive cycle to create best practices for the rest of the organization."
        )
        report_parts.append(positive_story)

        # --- The Vicious Cycle (Negative Story) ---
        report_parts.append(
            "\n\n---\n\n## The Vicious Cycle: Uncovering the Root Cause 🌡️\n\n"
            "Conversely, the weakest areas reveal a negative cycle where challenges in one domain exacerbate problems in another. This is the most critical area for intervention.\n"
        )
        negative_story = (
            f"The most significant challenge in the **{name_A}** survey is a lack of **{themes_A['lowest'].replace('_', ' ')}** "
            f"(metric: '{self.findings_A['lowest_metric']['name']}', score: {self.findings_A['lowest_metric']['score']:.2f}). "
            f"This issue appears to be directly linked to the weakness in **{themes_B['lowest'].replace('_', ' ')}** highlighted in the **{name_B}** survey "
            f"(metric: '{self.findings_B['lowest_metric']['name']}', score: {self.findings_B['lowest_metric']['score']:.2f}).\n\n"
            "**Insight**: This is not two separate problems; it is a single, interconnected challenge. For instance, a failure in our communication channels is creating an information vacuum, which in turn erodes trust in leadership. "
            "Solving the second problem is impossible without first addressing the first."
        )
        report_parts.append(negative_story)
        
        # --- Shared Demographics ---
        hotspots_A = {f"{spot['Function']} ({spot['Job Level']})" for spot in self.findings_A.get('hotspots', [])}
        hotspots_B = {f"{spot['Function']} ({spot['Job Level']})" for spot in self.findings_B.get('hotspots', [])}
        shared_hotspots = hotspots_A.intersection(hotspots_B)
        
        bright_spots_A = {f"{spot['Function']} ({spot['Job Level']})" for spot in self.findings_A.get('bright_spots', [])}
        bright_spots_B = {f"{spot['Function']} ({spot['Job Level']})" for spot in self.findings_B.get('bright_spots', [])}
        shared_bright_spots = bright_spots_A.intersection(bright_spots_B)
    
        if shared_hotspots or shared_bright_spots:
            report_parts.append("\n\n---\n\n## The Human Element: Who is Most Affected? 👥\n")
            if shared_bright_spots:
                report_parts.append(
                    f"**Shared Bright Spots**: The positive cycle mentioned above is strongest within the following groups: **{', '.join(list(shared_bright_spots))}**. "
                    "These teams are our internal champions and role models."
                )
            if shared_hotspots:
                report_parts.append(
                    f"\n**Shared Hotspots**: The negative cycle is hitting these groups the hardest: **{', '.join(list(shared_hotspots))}**. "
                    "They are experiencing a compounded failure across both drivers and require immediate, targeted support."
                )
        
        # --- Cross-driver: Function-level overlaps by THEME (new) ---
        # Map per-metric Function-level findings to their themes using each findings' metric_themes,
        # then look for overlaps where the same theme + Function appear in both drivers.
        def build_theme_function_set(findings, key):
            theme_map = findings.get('metric_themes', {}) or {}
            items = set()
            for spot in findings.get(key, []):
                metric = spot.get('Metric')
                func = spot.get('Function')
                if not metric or not func:
                    continue
                theme = theme_map.get(metric)
                if not theme:
                    continue
                items.add(f"{theme}||{func}")
            return items

        hotspots_theme_A = build_theme_function_set(self.findings_A, 'hotspots_by_driver')
        hotspots_theme_B = build_theme_function_set(self.findings_B, 'hotspots_by_driver')
        bright_theme_A = build_theme_function_set(self.findings_A, 'bright_spots_by_driver')
        bright_theme_B = build_theme_function_set(self.findings_B, 'bright_spots_by_driver')

        overlapping_hotspot_themes = hotspots_theme_A.intersection(hotspots_theme_B)
        overlapping_bright_themes = bright_theme_A.intersection(bright_theme_B)

        if overlapping_bright_themes or overlapping_hotspot_themes:
            report_parts.append("\n\n---\n\n## Cross-driver: Function-level Overlaps by Theme 🔍\n")
            if overlapping_bright_themes:
                report_parts.append("**Shared Brightness (same theme & function across both drivers):**\n")
                for item in sorted(overlapping_bright_themes):
                    theme, func = item.split("||")
                    report_parts.append(f"- Theme **{theme.replace('_', ' ')}** — Function **{func}** is a bright spot in both drivers.")
            if overlapping_hotspot_themes:
                report_parts.append("\n**Shared Pain (same theme & function across both drivers):**\n")
                for item in sorted(overlapping_hotspot_themes):
                    theme, func = item.split("||")
                    report_parts.append(f"- Theme **{theme.replace('_', ' ')}** — Function **{func}** is a hotspot in both drivers.")

        # --- Unified Recommendations ---
        report_parts.append(
            "\n\n---\n\n## Unified Strategic Recommendations 🎯\n\n"
            "To break the vicious cycle and amplify the reinforcing loop, a unified strategy is essential:\n"
        )
        rec1 = (
            f"**1. Address the Root Cause First**: The analysis shows the weakness in **{themes_B['lowest'].replace('_', ' ')}** "
            f"is a foundational problem affecting **{themes_A['lowest'].replace('_', ' ')}**. "
            "Therefore, the primary strategic focus should be on solving this root cause. Any effort to fix the symptom without addressing the cause will likely fail."
        )
        rec2 = (
            "**2. Learn from the Bright Spots**: Engage with the leaders and teams within the 'Shared Bright Spots' to understand their practices. "
            "Codify their methods into a 'playbook' that can be shared with the rest of the organization, especially with the leaders of the 'Shared Hotspots'."
        )
        rec3 = (
            "**3. Launch a Targeted Intervention for Hotspots**: Create a cross-functional task force to directly support the 'Shared Hotspot' groups. "
            "This should involve listening sessions to understand their specific barriers, followed by the co-creation of a tailored support plan. This demonstrates a commitment to solving the problem and ensures no group is left behind."
        )
        report_parts.append(f"- {rec1}\n- {rec2}\n- {rec3}")

        return "\n".join(report_parts)