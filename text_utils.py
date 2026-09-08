from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable


PLACEHOLDERS = {"", "na", "n/a", "-", "null", "none", "0", "nan"}

COMMERCIAL_PATTERNS = (
    re.compile(r"\b(?:qty|quantity)\s*[:=-]?\s*\d+(?:\.\d+)?\b", re.I),
    re.compile(r"\b(?:rs\.?|inr|₹)\s*\d[\d,.]*\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*%\s*gst\b", re.I),
    re.compile(r"\bgst\s*(?:@|[:=-])?\s*\d+(?:\.\d+)?\s*%?\b", re.I),
    re.compile(r"\bgst\b", re.I),
    re.compile(r"\b(?:delivery|del\.)\s*(?:date)?\s*[:=-]?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.I),
    re.compile(r"\bpack(?:ing)?\s*(?:of|size)?\s*[:=-]?\s*\d+\b", re.I),
)

UNIT_ALIASES = {
    "square millimetres": "mm2", "square millimetre": "mm2", "sq mm": "mm2", "sqmm": "mm2", "mm²": "mm2", "mm2": "mm2",
    "square centimetres": "cm2", "square centimetre": "cm2", "sq cm": "cm2", "sqcm": "cm2", "cm²": "cm2", "cm2": "cm2",
    "millimetres": "mm", "millimetre": "mm", "millimeters": "mm", "millimeter": "mm", "mm": "mm",
    "centimetres": "cm", "centimetre": "cm", "centimeters": "cm", "centimeter": "cm", "cm": "cm",
    "metres": "m", "metre": "m", "meters": "m", "meter": "m", "mtr": "m", "mtrs": "m",
    "kilograms": "kg", "kilogram": "kg", "kgs": "kg", "kg": "kg",
    "grams": "g", "gram": "g", "gms": "g", "gm": "g", "g": "g",
    "kilowatts": "kw", "kilowatt": "kw", "kw": "kw",
    "watts": "w", "watt": "w", "w": "w",
    "volts": "v", "volt": "v", "v": "v",
    "amperes": "a", "ampere": "a", "amps": "a", "amp": "a", "a": "a",
    "litres": "l", "litre": "l", "liters": "l", "liter": "l", "ltr": "l", "ltrs": "l", "l": "l",
    "inches": "in", "inch": "in", "in": "in",
    "feet": "ft", "foot": "ft", "ft": "ft",
    "bar": "bar", "bars": "bar",
    "hz": "hz"
}

UNIT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>" +
    "|".join(sorted((re.escape(k) for k in UNIT_ALIASES), key=len, reverse=True)) +
    r")(?![a-z])",
    re.I,
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[&+#][a-z0-9]+)*", re.I)
CODE_RE = re.compile(r"(?i)\b(?=[a-z0-9._/-]{4,}\b)(?=[a-z0-9._/-]*\d)[a-z0-9]+(?:[._/-][a-z0-9]+)*\b")
MODEL_CUE_RE = re.compile(
    r"(?i)\b(?:model|part|product|item)\s*(?:no\.?|number|code)\s*[:#=-]?\s*"
    r"(?P<code>[a-z0-9]+(?:[._/-][a-z0-9]+)*(?:\s+\d{1,6})?)\b"
)
SHORT_CODE_RE = re.compile(r"(?i)\b[a-z0-9]+(?:[._/-][a-z0-9]+)*\b")
HSN_RE = re.compile(r"(?i)\bhsn(?:\s+code)?\s*[:#=-]?\s*[a-z0-9._/-]+")
ERP_CUE_RE = re.compile(
    r"(?i)\b(?:erp(?:\s+code)?|sku|material\s+(?:id|code|number|no))"
    r"\s*[:#=-]?\s*(?P<code>[a-z0-9][a-z0-9._/-]{3,})\b"
)
SPECIFICATION_CODE_RE = re.compile(
    r"(?i)^\d+(?:\.\d+)?(?:mm2|cm2|sqmm|mm|cm|m|in|ft|a|amp|v|w|kw|g|kg|l|bar|hz|core|pin|pole|way)$"
)
PREFIXED_SPECIFICATION_CODE_RE = re.compile(
    r"(?i)^(?:(?:m|id|od|nb|awg|swg)\d+(?:x\d+(?:\.\d+)?){0,3}|(?:mm|cm)2)$"
)


def load_json_map(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {str(k).casefold(): str(v).casefold() for k, v in json.load(handle).items()}


def clean_scalar(value: object) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).strip()
    if text.casefold() in PLACEHOLDERS:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_commercial_noise(text: str) -> str:
    for pattern in COMMERCIAL_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_code(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", clean_scalar(value).casefold())


def split_erp_codes(value: object) -> list[str]:
    raw = clean_scalar(value)
    if not raw:
        return []
    parts = re.split(r"[;,|\s]+", raw)
    return sorted({compact_code(part) for part in parts if compact_code(part)})


def _normalize_units(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group("value")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        unit = UNIT_ALIASES[match.group("unit").casefold()]
        return f"{value}{unit}"

    return UNIT_PATTERN.sub(replace, text)


def _expand_abbreviations(text: str, abbreviations: dict[str, str]) -> str:
    if not abbreviations:
        return text
    return " ".join(abbreviations.get(token, token) for token in text.split())


def normalize_text(
    value: object,
    abbreviations: dict[str, str] | None = None,
    brand_aliases: dict[str, str] | None = None,
    remove_commercial_noise: bool = False,
) -> str:
    text = clean_scalar(value)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", unicodedata.normalize("NFKC", text))
    text = text.casefold()
    text = text.replace("×", "x")
    if remove_commercial_noise:
        text = strip_commercial_noise(text)
    text = _normalize_units(text)
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "x", text)
    text = re.sub(r"[-_/,;:()\[\]{}]+", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s&+#.]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    if brand_aliases and text in brand_aliases:
        text = brand_aliases[text]
    text = _expand_abbreviations(text, abbreviations or {})
    return re.sub(r"\s+", " ", text).strip()


def extract_numeric_tokens(text: object) -> list[str]:
    normalized = _normalize_units(unicodedata.normalize("NFKC", clean_scalar(text)).casefold())
    tokens: set[str] = set()
    for match in UNIT_PATTERN.finditer(normalized):
        value = match.group("value").rstrip("0").rstrip(".") if "." in match.group("value") else match.group("value")
        tokens.add(f"{value}{UNIT_ALIASES[match.group('unit').casefold()]}")
    for dims in re.findall(r"\b\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?){1,3}(?:mm|cm|m)?\b", normalized):
        tokens.add(dims.replace(" ", ""))
    return sorted(tokens)


def extract_model_tokens(text: object) -> list[str]:
    raw = unicodedata.normalize("NFKC", clean_scalar(text)).casefold()
    raw = HSN_RE.sub(" ", raw)
    numeric_tokens = {compact_code(token) for token in extract_numeric_tokens(raw)}
    output: set[str] = set()

    def add(candidate: str, explicit: bool = False) -> None:
        compact = compact_code(candidate)
        if explicit:
            minimum_length = 2
        elif any(character.isalpha() for character in compact):
            minimum_length = 3
        else:
            # Preserve V3's useful handling of long numeric bearing/model
            # series such as 6205 without promoting dimensions like 6-22.
            minimum_length = 4
        if len(compact) < minimum_length:
            return
        if not any(character.isdigit() for character in compact):
            return
        if not explicit and (
            compact in numeric_tokens
            or SPECIFICATION_CODE_RE.fullmatch(compact)
            or PREFIXED_SPECIFICATION_CODE_RE.fullmatch(compact)
        ):
            return
        output.add(compact)

    # Explicit model/part/product/item labels are authoritative. This keeps
    # short and numeric industrial identifiers such as 831, 937 and A 26.
    for match in MODEL_CUE_RE.finditer(raw):
        add(match.group("code"), explicit=True)

    # Preserve the original V3 behavior for longer codes such as 6205-ZZ and
    # A9F74106.
    for token in CODE_RE.findall(raw):
        add(token)

    # Also retain compact short alphanumeric series used heavily in belts,
    # bearings and tools: A52, B-41, C48, PH2, CTR16. Measurement-like M8,
    # ID08 and 20MM tokens remain excluded unless explicitly labelled.
    for token in SHORT_CODE_RE.findall(raw):
        compact = compact_code(token)
        if any(character.isalpha() for character in compact):
            add(token)
    return sorted(output)


def extract_code_candidates(text: object) -> list[str]:
    return extract_model_tokens(text)[:5]


def extract_trusted_erp_candidates(text: object) -> list[str]:
    """Return only ERP-like query values safe enough for the exact-code route.

    General model numbers, HSN codes and dimensional values intentionally do
    not qualify. Pure numeric ERP values require an explicit ERP/SKU/material
    code cue; an uncued code-only query must contain both letters and digits.
    """
    raw = unicodedata.normalize("NFKC", clean_scalar(text)).casefold()
    candidates: list[str] = []
    for match in ERP_CUE_RE.finditer(raw):
        compact = compact_code(match.group("code"))
        if len(compact) >= 4 and not SPECIFICATION_CODE_RE.fullmatch(compact):
            candidates.append(compact)

    stripped = raw.strip()
    if re.fullmatch(r"[a-z0-9._/-]{6,}", stripped):
        compact = compact_code(stripped)
        if (
            re.search(r"[a-z]", compact)
            and re.search(r"\d", compact)
            and not SPECIFICATION_CODE_RE.fullmatch(compact)
        ):
            candidates.append(compact)
    return list(dict.fromkeys(candidates))[:2]


def physical_fingerprint(*fields: object) -> str:
    canonical = "|".join(normalize_text(field) for field in fields)
    return hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()


def normalize_material_record(
    record: dict[str, object],
    abbreviations: dict[str, str],
    brand_aliases: dict[str, str],
    remove_commercial_noise: bool = True,
) -> dict[str, object]:
    raw = {name: clean_scalar(record.get(name, "")) for name in (
        "materialId", "categoryName", "brandName", "productName", "productSpecification", "companyERPCode"
    )}
    brand = normalize_text(raw["brandName"], brand_aliases=brand_aliases)
    category = normalize_text(raw["categoryName"], abbreviations=abbreviations)
    product = normalize_text(raw["productName"], abbreviations=abbreviations)
    specification = normalize_text(raw["productSpecification"], abbreviations=abbreviations)
    erp_codes = split_erp_codes(raw["companyERPCode"])
    combined_raw = " ".join((raw["brandName"], raw["categoryName"], raw["productName"], raw["productSpecification"], raw["companyERPCode"]))
    all_text = normalize_text(
        combined_raw,
        abbreviations=abbreviations,
        remove_commercial_noise=remove_commercial_noise,
    )
    embedding_text = " ".join(part for part in (brand, category, product, specification) if part)
    searchable_description = " ".join((raw["productName"], raw["productSpecification"]))
    model_tokens = extract_model_tokens(searchable_description)
    numeric_tokens = extract_numeric_tokens(searchable_description)
    return {
        "id": raw["materialId"],
        **raw,
        "source": clean_scalar(record.get("source", "")),
        "categoryNameNormalized": category,
        "brandNameNormalized": brand,
        "productNameNormalized": product,
        "productSpecificationNormalized": specification,
        "companyERPCodeNormalized": erp_codes,
        "modelTokens": model_tokens,
        "numericTokens": numeric_tokens,
        "allTextNormalized": all_text,
        "embeddingText": embedding_text,
        "physicalFingerprint": physical_fingerprint(brand, category, product, specification),
    }


def query_features(
    query: object,
    abbreviations: dict[str, str],
    brand_aliases: dict[str, str],
) -> dict[str, object]:
    raw = clean_scalar(query)
    return {
        "raw": raw,
        "normalized": normalize_text(raw, abbreviations=abbreviations, remove_commercial_noise=True),
        "code_candidates": extract_code_candidates(raw),
        "trusted_erp_candidates": extract_trusted_erp_candidates(raw),
        "model_tokens": extract_model_tokens(raw),
        "numeric_tokens": extract_numeric_tokens(raw),
    }


@lru_cache(maxsize=4096)
def unit_value(token: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]+)", token)
    return (match.group(1), match.group(2)) if match else None


def numeric_conflicts(query_tokens: Iterable[str], document_tokens: Iterable[str]) -> tuple[int, int]:
    query_by_unit: dict[str, set[str]] = {}
    doc_by_unit: dict[str, set[str]] = {}
    for token in query_tokens:
        parsed = unit_value(token)
        if parsed:
            query_by_unit.setdefault(parsed[1], set()).add(parsed[0])
    for token in document_tokens:
        parsed = unit_value(token)
        if parsed:
            doc_by_unit.setdefault(parsed[1], set()).add(parsed[0])
    matches = conflicts = 0
    for unit, values in query_by_unit.items():
        if unit not in doc_by_unit:
            continue
        if values & doc_by_unit[unit]:
            matches += 1
        else:
            conflicts += 1
    return matches, conflicts
