"""Licence plate text rules for Moldova and the foreign plates most often seen there.

The OCR model returns raw text such as ``"KCA123"``, ``"8LAB123"`` or ``"MDABC123"``.
:func:`interpret` turns it into a canonical plate by:

* normalising it (upper case, Cyrillic look-alikes to Latin, no spaces or dashes),
* fitting it to the known plate layouts, fixing the classic OCR swaps (0/O, 1/I,
  8/B, 5/S, ...) only where a layout says a letter or a digit must be,
* ranking the interpretations so the most plausible layout wins.

Canonical plates contain only ``A-Z`` and ``0-9``. The car wash system should store
registered plates in the same form; :func:`normalize` produces it from user input
such as ``"c ab 123"`` or ``"ABC-123"``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

# Mask symbols: "@" = any letter, "#" = any digit, " " = display separator.
# Any other character is a literal that must appear at that position.
LETTER = "@"
DIGIT = "#"

CATEGORIES = ("moldova", "moldova_special", "foreign", "unknown")

# Cyrillic letters that look like Latin ones (Transnistrian/Ukrainian plates, OCR slips).
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
        "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ї": "I", "Ј": "J",
        "Ѕ": "S",
    }
)
_NOT_ALNUM = re.compile(r"[^A-Z0-9]")

# Used only where the layout requires a letter but the OCR produced a digit (and vice versa).
_DIGIT_AS_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_LETTER_AS_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8",
    "G": "6", "T": "7", "A": "4",
}

# Every character the OCR got "wrong" for the chosen layout makes that layout less likely;
# needing more than MAX_FIXES changes means the text is not that layout at all.
FIX_PENALTY = 0.25
MAX_FIXES = 2
# Reading the "MD" country code of the plate's left band as part of the number.
MD_PREFIX_PENALTY = 0.9

# District codes of the pre-2015 plates ("BL AB 123"). Chișinău used a single C or K,
# which makes those plates look exactly like the current "ABC 123" layout.
MD_DISTRICTS = frozenset(
    "AN BE BL BR BS CC CG CH CL CM CN CO CR CS CT CU DB DN DR ED FL FR GE GL GR HN IL "
    "LP LV NS OC OR RB RS RZ SD SG SL SO SR ST SV TG TI TL TR UN VL".split()
)
RO_COUNTIES = frozenset(
    "AB AG AR BC BH BN BR BT BV BZ CJ CL CS CT CV DB DJ GJ GL GR HD HR IF IL IS MH MM MS "
    "NT OT PH SB SJ SM SV TL TM TR VL VN VS".split()
)
# Ukrainian plates only use letters that exist in both the Latin and Cyrillic alphabets.
UA_LETTERS = frozenset("ABCEHIKMOPTX")


@dataclass(frozen=True)
class PlateFormat:
    name: str
    country: str
    category: str
    masks: tuple[str, ...]
    prior: float
    description: str
    check: Callable[[str], bool] | None = None


FORMATS: tuple[PlateFormat, ...] = (
    PlateFormat(
        "md_standard", "MD", "moldova", ("@@@ ###",), 1.0,
        "Standard since 2015, e.g. KCA 123 (also green EV plates, RMG/RMP government plates "
        "and pre-2015 Chișinău C/K plates)",
    ),
    PlateFormat(
        "md_regional", "MD", "moldova", ("@@ @@ ###", "@@ @@ ##", "@@ @@ #"), 0.9,
        "Pre-2015 district plates, e.g. BL AB 123",
        check=lambda p: p[:2] in MD_DISTRICTS,
    ),
    PlateFormat(
        "md_short", "MD", "moldova", ("@@@ ##", "@@@ #"), 0.8,
        "Personalised plates with one or two digits, e.g. ION 7",
    ),
    PlateFormat(
        "md_transnistria", "MD", "moldova", ("@ ### @@",), 0.6,
        "Transnistrian region plates (and some trailer series), e.g. A 123 BC",
    ),
    PlateFormat("md_police", "MD", "moldova_special", ("MAI ####",), 0.6, "Ministry of Internal Affairs"),
    PlateFormat("md_president", "MD", "moldova_special", ("RM ####",), 0.5, "Presidential plates"),
    PlateFormat("md_military", "MD", "moldova_special", ("FA ####",), 0.5, "Armed forces"),
    PlateFormat(
        "md_diplomatic", "MD", "moldova_special", ("CD ### @@", "TC ### @@", "TS ### @@"), 0.5,
        "Diplomatic, consular and service staff plates",
    ),
    PlateFormat(
        "md_temporary", "MD", "moldova_special", ("H ####", "P ####", "T ####"), 0.5,
        "Temporary plates (foreign residents, temporary admission, export)",
    ),
    PlateFormat(
        "md_four_letter", "MD", "moldova_special", ("@@@@ ###", "@@@@ ##", "@@@@ #"), 0.3,
        "Four-letter series (emergency services, unlisted pre-2015 district codes)",
    ),
    PlateFormat(
        "ro_bucharest", "RO", "foreign", ("B ### @@@", "B ## @@@"), 0.6, "Romania, Bucharest",
    ),
    PlateFormat(
        "ro", "RO", "foreign", ("@@ ## @@@",), 0.6, "Romania, counties",
        check=lambda p: p[:2] in RO_COUNTIES,
    ),
    PlateFormat(
        "ua", "UA", "foreign", ("@@ #### @@",), 0.6, "Ukraine",
        check=lambda p: all(c in UA_LETTERS for c in p[:2] + p[-2:]),
    ),
)

FORMATS_BY_NAME = {f.name: f for f in FORMATS}


@dataclass(frozen=True)
class PlateInterpretation:
    plate: str  # canonical text, e.g. "ABC123"
    display: str  # human friendly, e.g. "ABC 123"
    format: str  # PlateFormat.name or "unknown"
    country: str  # "MD", "RO", "UA", or "" when unknown
    category: str  # one of CATEGORIES
    fixes: int  # characters changed to fit the layout
    score: float  # plausibility used to rank interpretations
    raw: str  # normalised OCR text before any fixes


def normalize(text: str) -> str:
    """Canonical plate text: upper case Latin letters and digits only."""
    text = unicodedata.normalize("NFKD", text.upper())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _NOT_ALNUM.sub("", text.translate(_CYRILLIC_TO_LATIN))


def _fit(text: str, mask: str) -> tuple[str, int] | None:
    """Fit text to a mask; returns (fixed text, number of fixes) or None."""
    slots = mask.replace(" ", "")
    if len(slots) != len(text):
        return None
    out = []
    fixes = 0
    for ch, slot in zip(text, slots):
        if slot == LETTER:
            if ch.isalpha():
                out.append(ch)
            elif ch in _DIGIT_AS_LETTER:
                out.append(_DIGIT_AS_LETTER[ch])
                fixes += 1
            else:
                return None
        elif slot == DIGIT:
            if ch.isdigit():
                out.append(ch)
            elif ch in _LETTER_AS_DIGIT:
                out.append(_LETTER_AS_DIGIT[ch])
                fixes += 1
            else:
                return None
        elif ch == slot:
            out.append(ch)
        elif _DIGIT_AS_LETTER.get(ch) == slot or _LETTER_AS_DIGIT.get(ch) == slot:
            out.append(slot)
            fixes += 1
        else:
            return None
        if fixes > MAX_FIXES:
            return None
    return "".join(out), fixes


def _display(plate: str, mask: str) -> str:
    out = []
    chars = iter(plate)
    for slot in mask:
        out.append(" " if slot == " " else next(chars))
    return "".join(out)


def candidates(text: str) -> list[PlateInterpretation]:
    """All layouts the text can be fitted to, most plausible first."""
    raw = normalize(text)
    variants = [(raw, 1.0)]
    # The "MD" under the flag on the plate's left band is sometimes read as part of the number.
    if raw.startswith("MD") and len(raw) >= 5:
        variants.append((raw[2:], MD_PREFIX_PENALTY))
    found = []
    for variant, variant_penalty in variants:
        for fmt in FORMATS:
            for mask in fmt.masks:
                fitted = _fit(variant, mask)
                if fitted is None:
                    continue
                plate, fixes = fitted
                if fmt.check is not None and not fmt.check(plate):
                    continue
                found.append(
                    PlateInterpretation(
                        plate=plate,
                        display=_display(plate, mask),
                        format=fmt.name,
                        country=fmt.country,
                        category=fmt.category,
                        fixes=fixes,
                        score=fmt.prior * variant_penalty * FIX_PENALTY**fixes,
                        raw=raw,
                    )
                )
    found.sort(key=lambda c: (-c.score, c.fixes))
    return found


def interpret(text: str) -> PlateInterpretation | None:
    """Best interpretation of OCR text, an "unknown" one if no layout fits, or None for noise."""
    found = candidates(text)
    if found:
        return found[0]
    raw = normalize(text)
    if not 2 <= len(raw) <= 10:
        return None
    return PlateInterpretation(
        plate=raw, display=raw, format="unknown", country="", category="unknown",
        fixes=0, score=0.05, raw=raw,
    )


def display(plate: str) -> str:
    """Human friendly spacing for a canonical plate ("BLAB123" -> "BL AB 123")."""
    best = interpret(plate)
    if best is None or best.plate != normalize(plate):
        return plate
    return best.display


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance; used to tell a misread of the same plate from another car."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
