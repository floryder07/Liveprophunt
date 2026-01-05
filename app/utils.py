from typing import Any, Dict, List


def parse_players(text: str) -> List[Dict[str, Any]]:
    """
    Parse a simple text format into player dictionaries.

    Format per line:
      name|stat_label|live_value|delta|target|pace[|team|money]

    - live_value may be int/float or empty.
    - target, pace converted to float when possible.
    - team and money are optional (only used when provided in the text area).
    Returns list of dicts with keys: name, stat_label, live_value, delta, target, pace, team, money
    """
    players: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        # Ensure length up to 8 (pad missing fields)
        while len(parts) < 8:
            parts.append("")
        # Unpack first 8 entries, extras are ignored
        name, stat_label, live_value, delta, target, pace, team, money = parts[:8]

        # parse live_value
        live_value_val = None
        if live_value != "":
            try:
                live_value_val = int(live_value)
            except ValueError:
                try:
                    live_value_val = float(live_value)
                except Exception:
                    # keep raw string (e.g., "N/A")
                    live_value_val = live_value or None

        # parse floats
        try:
            target_val = float(target) if target != "" else None
        except Exception:
            target_val = None
        try:
            pace_val = float(pace) if pace != "" else None
        except Exception:
            pace_val = None

        players.append(
            {
                "name": name or "Unknown",
                "stat_label": stat_label or "Stat",
                "live_value": live_value_val,
                "delta": delta or "",
                "target": target_val,
                "pace": pace_val,
                "team": team or None,
                "money": money or None,
            }
        )
    return players
