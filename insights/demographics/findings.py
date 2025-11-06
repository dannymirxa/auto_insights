def executive_summary(driver_name: str, average_score: float, num_metrics: int) -> str:
    return (
        f"## Executive Summary\n\n"
        f"The {driver_name} driver was analyzed across {num_metrics} metric(s). "
        f"Average score across available metrics: {average_score:.2f}."
    )

def key_observations(highest_metric: dict, lowest_metric: dict) -> list:
    """
    Returns list of markdown lines describing highest and lowest scoring metrics.
    """
    lines = []
    lines.append(f"- Highest scoring metric: **{highest_metric['name']}** ({highest_metric['score']:.2f}).")
    lines.append(f"- Lowest scoring metric: **{lowest_metric['name']}** ({lowest_metric['score']:.2f}).")
    return lines


def recommendations_from_theme(lowest_metric_name: str, metric_themes: dict, recommendation_map: dict) -> str:
    """
    Generate a recommendation sentence given the lowest metric name and mappings.
    """
    lowest_theme = metric_themes.get(lowest_metric_name, 'default')
    recommendation = recommendation_map.get(lowest_theme, recommendation_map.get('default', 'No recommendation available.'))
    return f"\n\n- The key area for improvement relates to **{lowest_theme.replace('_', ' ')}**. {recommendation}"


def pairwise_comparison_helpers(findings_A: dict, findings_B: dict) -> dict:
    """
    Extract shared hotspots/bright spots and themes for pairwise comparison report generation.
    Returns a dict with shared_hotspots, shared_bright_spots, themes_A, themes_B
    """
    def get_themes(findings):
        themes = findings.get('metric_themes', {})
        lowest_metric_name = findings['lowest_metric']['name']
        highest_metric_name = findings['highest_metric']['name']
        return {
            'lowest': themes.get(lowest_metric_name),
            'highest': themes.get(highest_metric_name)
        }

    themes_A = get_themes(findings_A)
    themes_B = get_themes(findings_B)

    hotspots_A = {f"{spot['Function']} ({spot['Job Level']})" for spot in findings_A.get('hotspots', [])}
    hotspots_B = {f"{spot['Function']} ({spot['Job Level']})" for spot in findings_B.get('hotspots', [])}
    shared_hotspots = hotspots_A.intersection(hotspots_B)

    bright_spots_A = {f"{spot['Function']} ({spot['Job Level']})" for spot in findings_A.get('bright_spots', [])}
    bright_spots_B = {f"{spot['Function']} ({spot['Job Level']})" for spot in findings_B.get('bright_spots', [])}
    shared_bright_spots = bright_spots_A.intersection(bright_spots_B)

    return {
        "themes_A": themes_A,
        "themes_B": themes_B,
        "shared_hotspots": shared_hotspots,
        "shared_bright_spots": shared_bright_spots,
    }