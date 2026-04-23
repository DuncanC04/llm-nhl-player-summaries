"""Shared natural-language prompt for table-to-text (all presets use the same user text)."""


def format_stats_text(stats):
    """Convert topStats array to readable text."""
    stats_text = []
    for stat in stats:
        stats_text.append(f"{stat['stat']}: {stat['value']} (percentile: {stat['pctl']})")
    return "; ".join(stats_text)


def create_prompt(example: dict) -> str:
    """Create a formatted prompt for training or inference."""
    stats_text = format_stats_text(example["topStats"])
    return f"""Generate a concise player summary based on the following information:

Name: {example['name']}
Team: {example['team']}
Position: {example['position']}
Top Statistics: {stats_text}

Summary:"""


def player_fields_prompt(name: str, team: str, position: str, top_stats: list) -> str:
    """Same layout as create_prompt but from scalar fields (generation path)."""
    stats_text = format_stats_text(top_stats)
    return f"""Generate a concise player summary based on the following information:

Name: {name}
Team: {team}
Position: {position}
Top Statistics: {stats_text}

Summary:"""
