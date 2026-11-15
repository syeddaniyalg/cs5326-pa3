import re
from decimal import Decimal, InvalidOperation

STEP_PATTERN = (
    r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)"
    r"(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
)
FINAL_ANSWER_PATTERN = re.compile(r"final\s*answer\s*(?:is\b|=)?\s*:?\s*\**\s*([^\n]*)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(
    r"^[\s\$\\\(\[\{\*_`~]*(?:boxed\s*\{)?[\s\$]*([-−]?)\s*\$?\s*(\d[\d,]*(?:\.\d+)?|\.\d+)"
)

SCIENTIFIC_SUFFIX = re.compile(r"[eE][+-]?\d")


def parse_steps(completion_text: str) -> list[str]:
    if not completion_text or not isinstance(completion_text, str):
        return []
    matches = re.findall(STEP_PATTERN, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for match in matches:
        text = match.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _to_integer(sign: str, digits: str) -> int | None:
    try:
        value = Decimal(digits.replace(",", ""))
    except InvalidOperation:
        return None
    if value != value.to_integral_value():
        return None
    integer = int(value)
    return -integer if sign else integer


def extract_answer(completion_text: str) -> int | None:
    if not completion_text or not isinstance(completion_text, str):
        return None
    try:
        matches = FINAL_ANSWER_PATTERN.findall(completion_text)
        if not matches:
            return None
        tail = matches[-1]
        number = NUMBER_PATTERN.match(tail)
        if number is None or SCIENTIFIC_SUFFIX.match(tail[number.end():]):
            return None
        return _to_integer(number.group(1), number.group(2))
    except Exception:
        return None
