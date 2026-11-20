import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.schemas.contracts import Account

FALLBACK_ABBREVIATIONS = {
    "a/p": "accounts payable",
    "a/r": "accounts receivable",
    "bldg": "building",
    "cam": "common area maintenance",
    "fx": "foreign exchange",
    "interco": "intercompany",
    "r&m": "repairs and maintenance",
    "ti": "tenant improvements",
    "wip": "work in progress",
}
FALLBACK_FLAGS = [
    "intercompany",
    "inter-co",
    "shareholder",
    "reimbursable",
    "entity",
    "foreign exchange",
    "suspense",
    "clearing",
    "unreconciled",
    "elimination",
    "loan",
    "escrow",
    "affiliate",
]


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def term_pattern(terms, suffix="") -> re.Pattern:
    alternatives = "|".join(
        re.escape(term) for term in sorted(terms, key=lambda term: (-len(term), term))
    )
    return re.compile(
        rf"(?<!\w)(?:{alternatives}){suffix}(?!\w)" if alternatives else r"(?!)"
    )


def load_rules(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fallback


@dataclass(frozen=True)
class Query:
    text: str
    normal_balance: str | None
    has_red_flag: bool
    explicit_balance: bool = False


class Normalizer:
    def __init__(self, data_dir: Path):
        self.abbreviations = {
            normalize(key): normalize(value)
            for key, value in load_rules(
                data_dir / "abbrev_map.json", FALLBACK_ABBREVIATIONS
            ).items()
        }
        self.abbreviation_pattern = term_pattern(self.abbreviations)
        flags = {
            normalize(term)
            for term in load_rules(data_dir / "red_flag_keywords.json", FALLBACK_FLAGS)
        }
        if "entity" in flags:
            flags.add("entities")
        self.flags = term_pattern(flags, suffix=r"(?:s|es)?")

    def expand(self, text: str) -> str:
        return self.abbreviation_pattern.sub(
            lambda match: self.abbreviations[match[0]], normalize(text)
        )

    def account_query(self, account: Account) -> Query:
        original = normalize(account.source_name or "")
        expanded = self.expand(original)
        balance = normalize(account.normal_balance or "")
        if balance in {"debit", "dr", "d"}:
            inferred = "Debit"
        elif balance in {"credit", "cr", "c"} or re.search(
            r"\b(accumulated depreciation|accounts payable|retained earnings)\b",
            expanded,
        ):
            inferred = "Credit"
        elif re.search(
            r"\b(accounts receivable|prepaid|cash|expense|expenses)\b", expanded
        ):
            inferred = "Debit"
        elif re.search(r"\b(revenue|income|liability|liabilities|equity)\b", expanded):
            inferred = "Credit"
        else:
            inferred = None
        return Query(
            expanded,
            inferred,
            bool(self.flags.search(original) or self.flags.search(expanded)),
            balance in {"debit", "dr", "d", "credit", "cr", "c"},
        )
