import pytest

from app.utils import parse_players


def test_parse_players_basic():
    text = "LeBron James|Points|22|+2.5 vs Line|19.5|28.1\nKevin Durant|Rebounds|6|-1.5 vs Line|7.5|6.8"
    players = parse_players(text)
    assert len(players) == 2
    assert players[0]["name"] == "LeBron James"
    assert players[0]["stat_label"] == "Points"
    assert players[0]["live_value"] == 22
    assert players[0]["delta"].startswith("+2.5")
    assert players[0]["target"] == pytest.approx(19.5)
    assert players[1]["name"] == "Kevin Durant"
    assert players[1]["live_value"] == 6


def test_parse_players_missing_fields():
    # Missing some columns; parser should not crash and should fill defaults
    text = "Player One|Assists|||"
    players = parse_players(text)
    assert len(players) == 1
    assert players[0]["name"] == "Player One"
    assert players[0]["stat_label"] == "Assists"
    assert players[0]["live_value"] is None
    assert players[0]["target"] is None
    assert players[0]["pace"] is None


def test_parse_players_floats_and_strings():
    text = "P|S|12.5|+1.0|10.0|5.5\nP2|S2|N/A|N/A|N/A|"
    players = parse_players(text)
    assert players[0]["live_value"] == pytest.approx(12.5)
    assert players[1]["live_value"] == "N/A" or players[1]["live_value"] is None
