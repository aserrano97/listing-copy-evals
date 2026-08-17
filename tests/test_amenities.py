from listing_gen.amenities import humanize_amenities, humanize_amenity, unknown_codes


def test_known_codes_use_curated_names() -> None:
    assert humanize_amenity("DishWasher") == "Dishwasher"
    assert humanize_amenity("InternetBroadband") == "Broadband internet (WiFi)"
    assert humanize_amenity("BathroomAndLaundry") == "Bathroom & laundry"


def test_unknown_codes_fall_back_to_decamelcase() -> None:
    assert humanize_amenity("SaunaPrivate") == "Sauna private"
    assert humanize_amenity("GameConsolePs5") == "Game console ps5"
    assert humanize_amenity("EvCharger") == "Ev charger"


def test_unknown_codes_are_flagged() -> None:
    codes = ["DishWasher", "SaunaPrivate", "EvCharger"]
    assert unknown_codes(codes) == ["SaunaPrivate", "EvCharger"]
    assert humanize_amenities(codes) == ["Dishwasher", "Sauna private", "Ev charger"]


def test_degenerate_code_survives() -> None:
    assert humanize_amenity("") == ""
    assert humanize_amenity("wifi") == "Wifi"
