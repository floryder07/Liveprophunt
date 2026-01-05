from typing import Any, Dict, List


def parse_players(text: str) -> List[Dict[str, Any]]:
    """
    Parse a simple text format into player dictionaries.

    Format per line:
      name|stat_label|live_value|delta|target|pace

    live_value may be int/float or empty. target and pace will be converted to float when possible.
    Returns list of dicts with keys: name, stat_label, live_value, delta, target, pace
    """
    players: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        # Ensure length
        while len(parts) < 6:
            parts.append("")
        name, stat_label, live_value, delta, target, pace = parts[:6]
        # parse live_value
        live_value_val = None
        if live_value != "":
            try:
                live_value_val = int(live_value)
            except ValueError:
                try:
                    live_value_val = float(live_value)
                except Exception:
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
            }
        )
    return players
