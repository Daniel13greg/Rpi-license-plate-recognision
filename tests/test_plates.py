import pytest

from carwash_lpr import plates


@pytest.mark.parametrize(
    "text, expected",
    [
        ("c ab 123", "CAB123"),
        ("ABC-123", "ABC123"),
        (" kca.123 ", "KCA123"),
        ("А123ВС", "A123BC"),  # Cyrillic look-alikes
        ("ȘȚĂ 12", "STA12"),  # Romanian diacritics typed by a cashier
        ("MD | KCA 123", "MDKCA123"),
    ],
)
def test_normalize(text, expected):
    assert plates.normalize(text) == expected


@pytest.mark.parametrize(
    "raw, plate, display, fmt, fixes",
    [
        # current standard and its OCR slips
        ("KCA123", "KCA123", "KCA 123", "md_standard", 0),
        ("kca 123", "KCA123", "KCA 123", "md_standard", 0),
        ("KCA12B", "KCA128", "KCA 128", "md_standard", 1),
        ("0IH812", "OIH812", "OIH 812", "md_standard", 1),
        ("RMG001", "RMG001", "RMG 001", "md_standard", 0),
        # "MD" from the plate's flag band read as part of the number
        ("MDKCA123", "KCA123", "KCA 123", "md_standard", 0),
        ("MDA123", "MDA123", "MDA 123", "md_standard", 0),
        # personalised plates with fewer digits
        ("ION7", "ION7", "ION 7", "md_short", 0),
        ("ALA01", "ALA01", "ALA 01", "md_short", 0),
        # pre-2015 district plates
        ("BLAB123", "BLAB123", "BL AB 123", "md_regional", 0),
        ("8LAB123", "BLAB123", "BL AB 123", "md_regional", 1),
        ("ANAB12", "ANAB12", "AN AB 12", "md_regional", 0),
        ("ANA812", "ANA812", "ANA 812", "md_standard", 0),
        # special series
        ("MAI1234", "MAI1234", "MAI 1234", "md_police", 0),
        ("MA11234", "MAI1234", "MAI 1234", "md_police", 1),
        ("RM0001", "RM0001", "RM 0001", "md_president", 0),
        ("FA1234", "FA1234", "FA 1234", "md_military", 0),
        ("CD123AB", "CD123AB", "CD 123 AB", "md_diplomatic", 0),
        ("H1234", "H1234", "H 1234", "md_temporary", 0),
        ("А123ВС", "A123BC", "A 123 BC", "md_transnistria", 0),
        # foreign plates common in Moldova
        ("B123ABC", "B123ABC", "B 123 ABC", "ro_bucharest", 0),
        ("B12ABC", "B12ABC", "B 12 ABC", "ro_bucharest", 0),
        ("CJ12ABC", "CJ12ABC", "CJ 12 ABC", "ro", 0),
        ("AA1234BB", "AA1234BB", "AA 1234 BB", "ua", 0),
    ],
)
def test_interpret_known_layouts(raw, plate, display, fmt, fixes):
    result = plates.interpret(raw)
    assert result is not None
    assert (result.plate, result.display, result.format, result.fixes) == (plate, display, fmt, fixes)


def test_categories_and_countries():
    assert plates.interpret("KCA123").category == "moldova"
    assert plates.interpret("KCA123").country == "MD"
    assert plates.interpret("MAI1234").category == "moldova_special"
    assert plates.interpret("CJ12ABC").category == "foreign"
    assert plates.interpret("CJ12ABC").country == "RO"
    assert plates.interpret("AA1234BB").country == "UA"


@pytest.mark.parametrize("raw", ["XX12ABC", "QQ1234ZZ", "STOP", "123456"])
def test_unmatched_text_is_unknown(raw):
    result = plates.interpret(raw)
    assert result is not None
    assert result.format == "unknown"
    assert result.category == "unknown"
    assert result.plate == plates.normalize(raw)


@pytest.mark.parametrize("raw", ["", "A", "-", "ABCDEFGHIJK"])
def test_noise_is_rejected(raw):
    assert plates.interpret(raw) is None


def test_too_many_fixes_is_not_a_layout_match():
    # Fitting this to the standard layout would need four letter<->digit swaps ("IZS488").
    assert plates.interpret("125A8B").format == "unknown"


def test_candidates_are_ranked():
    found = plates.candidates("BLAB12")
    assert found[0].format == "md_regional"
    assert any(c.format == "md_standard" and c.plate == "BLA812" for c in found[1:])


def test_display():
    assert plates.display("BLAB123") == "BL AB 123"
    assert plates.display("KCA123") == "KCA 123"
    assert plates.display("XYZ") == "XYZ"
    # an OCR fix would change the text, so the input is shown as-is
    assert plates.display("KCA12B") == "KCA12B"


@pytest.mark.parametrize(
    "a, b, distance",
    [("ABC123", "ABC123", 0), ("ABC123", "ABC128", 1), ("ABC123", "BC123", 1), ("ABC123", "XYZ789", 6), ("", "AB", 2)],
)
def test_edit_distance(a, b, distance):
    assert plates.edit_distance(a, b) == distance
    assert plates.edit_distance(b, a) == distance
