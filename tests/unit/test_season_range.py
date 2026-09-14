import pytest

from nfl_predict.data.season_range import parse_season_range


def test_single_year():
    assert parse_season_range("2025") == [2025]


def test_hyphenated_range():
    assert parse_season_range("2010-2013") == [2010, 2011, 2012, 2013]


def test_comma_separated_list():
    assert parse_season_range("2023,2024,2025") == [2023, 2024, 2025]


def test_mixed_ranges_and_singles_deduplicated_and_sorted():
    assert parse_season_range("2020,2010-2012,2011") == [2010, 2011, 2012, 2020]


def test_whitespace_is_tolerated():
    assert parse_season_range(" 2010 - 2012 , 2020 ") == [2010, 2011, 2012, 2020]


def test_empty_spec_raises():
    with pytest.raises(ValueError):
        parse_season_range("")


def test_reversed_range_raises():
    with pytest.raises(ValueError):
        parse_season_range("2025-2010")


def test_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_season_range("twenty-ten")


def test_implausible_year_raises():
    with pytest.raises(ValueError):
        parse_season_range("1500")
