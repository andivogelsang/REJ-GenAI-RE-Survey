import html
import re


# --- Label cleaning helper ---
def clean_label(label: str) -> str:
    """Unescape HTML entities and replace '&' with 'and'."""
    if not isinstance(label, str):
        return label
    text = html.unescape(label)        # e.g., &amp; -> &
    return text.replace('&', 'and')    # e.g., & -> and


# --- Label extractors using the cleaner ---
def label_from_brackets(col: str) -> str:
    m = re.search(r'\[(.*?)\]', col)
    text = m.group(1).strip() if m else col
    return clean_label(text)


def label_strip_brackets_and_parens(col: str) -> str:
    base = col.split(']')[0].split('[')[-1].strip()
    base = re.sub(r"\s*\(.*?\)", "", base).strip()
    return clean_label(base)


def extract_task_name(col: str) -> str:
    m = re.search(r'\[(.*?)\]', col)
    task = m.group(1) if m else col
    task = re.sub(r'\(.*?\)', '', task).strip()
    return clean_label(task)
