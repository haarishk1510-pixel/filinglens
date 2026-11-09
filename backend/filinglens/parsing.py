"""Turn 10-K HTML into section-aware text chunks, keeping tables intact."""
import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# Filings are XHTML with inline XBRL; lxml's HTML parser handles them correctly.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Market Risk Disclosures",
    "8": "Financial Statements",
    "9": "Accounting Disagreements",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions Disclosure",
    "10": "Directors and Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Related Transactions",
    "14": "Accountant Fees",
    "15": "Exhibits",
    "16": "Form 10-K Summary",
}

SECTION_NAMES_20F = {
    "1": "Directors and Advisers",
    "2": "Offer Statistics",
    "3": "Key Information and Risk Factors",
    "4": "Information on the Company",
    "4A": "Unresolved Staff Comments",
    "5": "Operating and Financial Review",
    "6": "Directors and Employees",
    "7": "Shareholders and Related Party Transactions",
    "8": "Financial Information",
    "9": "The Offer and Listing",
    "10": "Additional Information",
    "11": "Market Risk Disclosures",
    "12": "Other Securities",
    "13": "Defaults and Delinquencies",
    "14": "Modifications to Rights",
    "15": "Controls and Procedures",
    "16": "Reserved",
    "16A": "Audit Committee Financial Expert",
    "16B": "Code of Ethics",
    "16C": "Accountant Fees",
    "16D": "Listing Standards Exemptions",
    "16E": "Share Purchases",
    "16F": "Change in Accountant",
    "16G": "Corporate Governance",
    "16H": "Mine Safety Disclosure",
    "16I": "Foreign Jurisdictions Disclosure",
    "16J": "Insider Trading Policies",
    "16K": "Cybersecurity",
    "17": "Financial Statements",
    "18": "Financial Statements",
    "19": "Exhibits",
}

# Case-sensitive on purpose: a heading is "Item 1A." or "ITEM 1A." and what
# follows starts with a capital, which rules out "Item 1A of this report".
HEADING = re.compile(r"^(?:Item|ITEM)\s+(\d{1,2}[A-K]?)(?![A-Za-z0-9])\s*[.:\-–—]?\s*([A-Z].*)?$")
MAX_HEADING_LENGTH = 200

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "center", "div", "dl", "h1",
    "h2", "h3", "h4", "h5", "h6", "li", "ol", "p", "section", "table", "ul",
}

MAX_CHUNK_WORDS = 350
OVERLAP_WORDS = 50
MAX_TABLE_WORDS = 450
MAX_TITLE_BLOCKS = 3
MAX_TITLE_WORDS = 12
MAX_CAPTION_WORDS = 40


@dataclass
class Block:
    kind: str  # "text" or "table"
    text: str


@dataclass
class Chunk:
    section: str
    kind: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _table_rows(table: Tag) -> list[str]:
    rows = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for td in tr.find_all(["td", "th"]):
            value = _clean(td.get_text(""))
            if not value:
                continue
            # Filings put "$" and ")" in their own cells; rejoin them to the number.
            if cells and cells[-1] in {"$", "€", "£"}:
                cells[-1] += value
            elif cells and value in {")", "%", ")%"}:
                cells[-1] += value
            else:
                cells.append(value)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _has_block_child(tag: Tag) -> bool:
    return any(tag.find(name) is not None for name in BLOCK_TAGS)


def _walk(node: Tag):
    for child in node.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if text and type(child) is NavigableString:
                yield Block("text", text)
        elif isinstance(child, Tag):
            if child.name == "table":
                rows = _table_rows(child)
                if rows:
                    yield Block("table", "\n".join(rows))
            elif _has_block_child(child):
                yield from _walk(child)
            else:
                text = _clean(child.get_text(""))
                if text:
                    yield Block("text", text)


def html_to_blocks(html: bytes | str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")
    return list(_walk(soup.body or soup))


def _section_from(block: Block, names: dict[str, str]) -> str | None:
    if block.kind == "table":
        lines = block.text.split("\n")
        # A table with many Item lines is the table of contents, not a heading.
        if len(lines) > 2:
            return None
        candidate = _clean(block.text.replace("|", " "))
    else:
        candidate = block.text

    if len(candidate) > MAX_HEADING_LENGTH:
        return None
    match = HEADING.match(candidate)
    if not match or match.group(1) not in names:
        return None
    return f"Item {match.group(1)}. {names[match.group(1)]}"


def _split_table(text: str) -> list[str]:
    """Split long tables by row, repeating the header row in every piece."""
    if len(text.split()) <= MAX_TABLE_WORDS:
        return [text]

    header, *rows = text.split("\n")
    pieces, current, words = [], [], 0
    for row in rows:
        row_words = len(row.split())
        if current and words + row_words > MAX_TABLE_WORDS:
            pieces.append("\n".join([header, *current]))
            current, words = [], 0
        current.append(row)
        words += row_words
    if current:
        pieces.append("\n".join([header, *current]))
    return pieces


def _chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks, current, words = [], [], 0
    for paragraph in paragraphs:
        tokens = paragraph.split()
        # A single paragraph longer than a chunk is split on word boundaries.
        step = MAX_CHUNK_WORDS - OVERLAP_WORDS
        parts = (
            [paragraph]
            if len(tokens) <= MAX_CHUNK_WORDS
            else [" ".join(tokens[i : i + MAX_CHUNK_WORDS]) for i in range(0, len(tokens), step)]
        )
        for part in parts:
            part_words = len(part.split())
            if current and words + part_words > MAX_CHUNK_WORDS:
                chunks.append("\n\n".join(current))
                tail = current[-1]
                keep_tail = len(tail.split()) <= OVERLAP_WORDS
                current = [tail] if keep_tail else []
                words = len(tail.split()) if keep_tail else 0
            current.append(part)
            words += part_words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _looks_like_title(text: str) -> bool:
    return (
        len(text.split()) <= MAX_TITLE_WORDS
        and not text.endswith(".")
        and not text.replace(",", "").isdigit()
    )


def _table_title(preceding_text: list[str]) -> str | None:
    """Find what names a table: a lead-in sentence or the short headings just above it."""
    if not preceding_text:
        return None
    last = preceding_text[-1]
    if last.endswith(":") and len(last.split()) <= MAX_CAPTION_WORDS:
        return last

    parts: list[str] = []
    for text in reversed(preceding_text[-MAX_TITLE_BLOCKS:]):
        if not _looks_like_title(text):
            break
        parts.insert(0, text)
    return " ".join(parts) or None


def blocks_to_chunks(blocks: list[Block], names: dict[str, str] = SECTION_NAMES) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "Cover and Introduction"
    paragraphs: list[str] = []
    preceding_text: list[str] = []

    def flush():
        nonlocal paragraphs
        chunks.extend(Chunk(section, "text", c) for c in _chunk_paragraphs(paragraphs))
        paragraphs = []

    for block in blocks:
        new_section = _section_from(block, names)
        if new_section:
            flush()
            section = new_section
            preceding_text = []
        elif block.kind == "table":
            flush()
            title = _table_title(preceding_text)
            prefix = f"Table: {title}\n" if title else ""
            chunks.extend(Chunk(section, "table", prefix + t) for t in _split_table(block.text))
            preceding_text = []
        else:
            paragraphs.append(block.text)
            preceding_text.append(block.text)
    flush()
    return chunks


def parse_filing(html: bytes | str, form: str = "10-K") -> list[Chunk]:
    return blocks_to_chunks(html_to_blocks(html), SECTION_NAMES_20F if form == "20-F" else SECTION_NAMES)
