from typing_extensions import List

ANALYSIS_CONFIG = [
    {
        'driver_name': "Vision and Direction",
        'metric_columns': ["adb|vision_lv1", "adb|vision_agree_lv1", "adb|conf_lv1_ldr", "enb|awareness", "adb|understand_purpose"]
    },
    {
        'driver_name': "Communication",
        'metric_columns': ["adb|info_managers", "adb|info_written", "ada|info_intranet"]
    },
    {
        'driver_name': "Business Leadership",
        'metric_columns': ["enb|ldr_support_system", "enb|ldr_time_resources", "rsb|quick_remedial", "sfb|current_change_mgmt", "enb|conf_lv2_ldr"]
    }
]

def get_columns_by_driver(drivers: List[str], config: List[dict]) -> List[str]:
    # Combine metric columns for all specified drivers.
    # Unknown drivers are ignored. Duplicates are removed while preserving order.
    index = {item["driver_name"]: item["metric_columns"] for item in config}

    combined: List[str] = []
    for driver in drivers:
        cols = index.get(driver)
        if not cols:
            continue
        for col in cols:
            if col not in combined:
                combined.append(col)
    return combined

print(get_columns_by_driver(["Vision and Direction", "Communication"], ANALYSIS_CONFIG))
