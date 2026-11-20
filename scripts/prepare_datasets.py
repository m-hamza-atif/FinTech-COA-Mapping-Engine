"""Prepare the target COA, training examples, test cases, and retrieval rules."""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval_text import code_key, hierarchy_text

ARTIFACTS = "Trial Balance Artifacts"
COA_FILE = "1.0 ChartofAccounts10_19_2026.xlsx"
MAPPING_FILE = "COA and GL Mapping.xlsx"
ENTITY_2 = "Trial Balance - Entity 2 PMC TB Mapped to Entity GL.xlsx"
PMC_DIR = "entity 1 - 10 mappings"
TRAIN_PMCS = (
    "1000101",
    "22833rd",
    "45724",
    "c0403901",
    "capcs133",
    "on_100lo",
    "srec132",
)
TEST_PMCS = ("slfals1", "223", "c2720")
ABBREVIATIONS = {
    "acq": "acquisition",
    "accum": "accumulated",
    "acct": "account",
    "adj": "adjustment",
    "adm": "administration",
    "admin": "administration",
    "amort": "amortization",
    "a/p": "accounts payable",
    "a/r": "accounts receivable",
    "bldg": "building",
    "cam": "common area maintenance",
    "capex": "capital expenditure",
    "cdn": "Canadian",
    "cpp": "Canada Pension Plan",
    "deprec": "depreciation",
    "depr": "depreciation",
    "dfc": "deferred financing costs",
    "disb": "disbursement",
    "eht": "employer health tax",
    "ei": "employment insurance",
    "equip": "equipment",
    "exp": "expense",
    "fx": "foreign exchange",
    "fmv": "fair market value",
    "gst": "goods and services tax",
    "hst": "harmonized sales tax",
    "hvac": "heating ventilation and air conditioning",
    "impr": "improvements",
    "ins": "insurance",
    "interco": "intercompany",
    "jv": "joint venture",
    "maint": "maintenance",
    "mgmt": "management",
    "misc": "miscellaneous",
    "nsf": "non-sufficient funds",
    "prop": "property",
    "py": "prior year",
    "r&m": "repairs and maintenance",
    "reimb": "reimbursement",
    "reit": "real estate investment trust",
    "reno": "renovation",
    "repl res": "replacement reserve",
    "rev": "revenue",
    "srvcs": "services",
    "svc": "services",
    "ti": "tenant improvements",
    "util": "utilities",
    "utl": "utilities",
    "vat": "value added tax",
    "wip": "work in progress",
}
# Reviewed against the team workbook and the ledger descriptions. Keep
# multi-meaning aliases scoped to the phrases that establish their meaning.
TEAM_ABBREVIATIONS = {
    "ar": "accounts receivable",
    "ap - cf": "accounts payable - cf",
    "ap - other": "accounts payable - other",
    "ap other": "accounts payable other",
    "acctng": "accounting",
    "equit": "equity",
    "distr": "distribution",
    "distrib": "distribution",
    "cad": "canadian dollar",
    "usd": "united states dollar",
    "eur": "euro",
    "gbp": "british pound sterling",
    "nok": "norwegian krone",
    "cap ex": "capital expenditure",
    "op fund": "operating fund",
    "prof fees": "professional fees",
    "svc chrg": "service charge",
    "mtm": "mark to market",
    "mkt leases": "market leases",
    "abv-blw mkt": "above and below market",
    "mkt-adm": "marketing administration",
    "mkt-events": "marketing events",
    "mkt-gen": "general marketing",
    "mkt-guest": "marketing guest",
    "mkt-hard costs": "marketing hard costs",
    "mkt-publicity": "marketing publicity",
    "int rate": "interest rate",
    "int exp": "interest expense",
    "int expense": "interest expense",
    "int. expense": "interest expense",
    "accrued int": "accrued interest",
    "deferred int": "deferred interest",
}
TRAP_TERMS = (
    "intercompany",
    "shareholder",
    "reimbursable",
    "entity",
    "fx",
    "foreign",
    "suspense",
    "clearing",
    "unreconciled",
    "elimination",
    "loan",
    "escrow",
    "affiliate",
    "restricted cash",
    "related party",
    "due to",
    "due from",
    "reclassification",
    "security deposit",
    "prepaid",
    "accrual",
    "inter-co",
    "interco",
    "functional currency",
    "reporting currency",
    "multiple currencies",
    "equity pickup",
    "equity accounted",
    "fair value",
    "right of use",
    "capital lease",
    "withholding tax",
    "unbilled",
    "other receivable",
    "capital contribution",
    "unitholder",
    "consolidation",
)


def clean(value):
    if value is None:
        return None
    return str(value).strip() or None


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def target_code(value):
    code = clean(value)
    key = code_key(code)
    if key and re.fullmatch(r"[0-9]{8}", key):
        return f"{key[:4]}-{key[4:]}"
    return code


def rows(path, sheet, first_row=1):
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        yield from workbook[sheet].iter_rows(min_row=first_row, values_only=True)
    finally:
        workbook.close()


def resolve_artifacts(raw_dir, extracted_dir):
    required = [Path(ENTITY_2)]
    required.extend(Path(PMC_DIR) / f"{name}.xlsx" for name in TRAIN_PMCS + TEST_PMCS)
    for root in (raw_dir / ARTIFACTS, extracted_dir / ARTIFACTS):
        if all((root / path).is_file() for path in required):
            return root

    archive_path = raw_dir / f"{ARTIFACTS}.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Supply {archive_path} or the extracted {ARTIFACTS} folder."
        )
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (extracted_dir / member.filename).resolve()
            if not destination.is_relative_to(extracted_dir.resolve()):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(extracted_dir)
    return extracted_dir / ARTIFACTS


def coa_rows(artifacts, mapping_file):
    primary = artifacts / COA_FILE
    if primary.is_file():
        return [
            row
            for row in rows(primary, "Report1", 7)
            if row[3] in {"Head", "Reg", "Tot"}
        ]
    return [
        row[2:]
        for row in rows(mapping_file, "Target COA", 8)
        if row[5] in {"Head", "Reg", "Tot"}
    ]


def industry_bucket(code, path, report_type):
    section = int(code.split("-")[0])
    if report_type == "Bal":
        if section < 2000:
            if any(
                re.search(
                    r"\b(capex|work in progress|under construction|capitalized costs)\b",
                    name,
                    re.IGNORECASE,
                )
                for name in path
            ):
                return "CapEx / Other"
            return "Assets"
        return "Liabilities" if section < 3000 else "Equity"
    if 4000 <= section < 5000 and section not in {4210, 4950}:
        return "Revenue"
    if section == 4210 or 5000 <= section < 6800:
        return "Operating Expense"
    return "CapEx / Other"


def build_target_coa(source):
    # Some heading links point past their own subtotal or are blank. A named
    # subtotal gives the heading its actual boundary in both cases.
    totals = {
        target_code(row[0]): target_code(row[10]) for row in source if row[3] == "Tot"
    }
    totals_by_name = defaultdict(list)
    for row in source:
        if row[3] == "Tot":
            totals_by_name[normalized(row[1]).removeprefix("total ")].append(
                target_code(row[0])
            )

    active = []
    result = {}
    for row in source:
        code = target_code(row[0])
        name = clean(row[1])
        kind = row[3]
        total = target_code(row[10])
        active = [heading for heading in active if code <= heading[1]]
        if kind == "Head":
            closing = next(
                (end for end in totals_by_name[normalized(name)] if end > code), total
            )
            if closing and closing > code:
                active.append((code, closing, name))
            continue
        if kind == "Tot":
            active = [heading for heading in active if heading[1] != code]
            continue

        # Follow the subtotal chain to distinguish overlapping sibling ranges.
        chain = set()
        while total and total not in chain:
            chain.add(total)
            total = totals.get(total)
        linked = [heading for heading in active if heading[1] in chain]
        anchor = linked[-1] if linked else None
        parents = [
            heading[2]
            for heading in active
            if heading in linked
            or anchor is None
            or (heading[0] < anchor[0] and heading[1] >= anchor[1])
        ]
        path = " > ".join(parents)
        report = {"Bal": "Bal", "Inc": "P&L", "P&L": "P&L"}[row[4]]
        balance = row[2]
        if not name or balance not in {"Debit", "Credit"} or code in result:
            raise ValueError(f"Invalid or duplicate postable COA account: {code}")
        result[code] = {
            "target_code": code,
            "target_name": name,
            "parent_path": path,
            "normal_balance": balance,
            "report_type": report,
            "industry_bucket": industry_bucket(code, [*parents, name], report),
            "indexed_text": hierarchy_text(name, path),
        }
    return result


def pmc_rows(artifacts, name):
    sheet = rows(artifacts / PMC_DIR / f"{name}.xlsx", "GLAccount_Template")
    headers = next(sheet)
    for cells in sheet:
        row = dict(zip(headers, cells))
        if not clean(row["SourceAccountCode"]):
            continue
        source_name = clean(row["SourceAccountName"])
        yield {
            "source_code": clean(row["SourceAccountCode"]),
            "source_name": source_name,
            "normal_balance": None,
            "ending_balance": None,
            "exclude": not source_name or clean(row.get("Exclude")) in {"1", "True"},
            "target_code": target_code(row["TargetAccountCode"]),
            "review_note": None,
        }


def reviewed_rows(path):
    balances = {
        clean(row[0]): row[2] for row in rows(path, "Source COA", 7) if clean(row[0])
    }
    sheet = rows(path, "Mapping Final")
    headers = list(next(sheet))
    comment_columns = [
        index for index, name in enumerate(headers) if "comment" in normalized(name)
    ]
    for cells in sheet:
        row = dict(zip(headers, cells))
        code = clean(row["Source Company orignal GL#"])
        if not code:
            continue
        name = clean(row["Source Company GL Description"])
        original = target_code(row["ERP System GL#"])
        proposed = target_code(row["Updated ERP System GL# proposed"])
        yield {
            "source_code": code,
            "source_name": name,
            "normal_balance": balances.get(code),
            "ending_balance": None,
            "exclude": not name,
            "target_code": proposed,
            "original_code": original,
            "changed": bool(proposed and code_key(proposed) != code_key(original)),
            "review_note": " | ".join(
                value for i in comment_columns if (value := clean(cells[i]))
            )
            or None,
        }


def stratified_sample(corrected, coa, size=100):
    strata = defaultdict(list)
    for row in corrected:
        account = coa.get(row["target_code"])
        bucket = account["industry_bucket"] if account else "Non-postable"
        strata[bucket].append(row)
    if len(corrected) < size:
        raise ValueError(f"Found {len(corrected)} corrected mappings, need {size}.")
    names = sorted(strata)
    quotas = {name: divmod(len(strata[name]) * size, len(corrected)) for name in names}
    extra = size - sum(count for count, _ in quotas.values())
    bonus = set(sorted(names, key=lambda name: -quotas[name][1])[:extra])
    rng = random.Random(42)
    return [
        row
        for name in names
        for row in rng.sample(strata[name], quotas[name][0] + (name in bonus))
    ]


def pair_key(row):
    return normalized(row["source_name"]), code_key(row["target_code"])


def build_examples(pmc, reviewed, coa):
    corrected = [row for row in reviewed if row["changed"]]
    sample = stratified_sample(corrected, coa)
    held_out = [row for name in TEST_PMCS for row in pmc[name]] + sample
    held_out_accounts = {
        (row["source_code"], normalized(row["source_name"])) for row in held_out
    }
    held_out_pairs = {pair_key(row) for row in held_out}
    training = {}
    training_rows = [(row, False) for row in reviewed]
    training_rows.extend((row, True) for name in TRAIN_PMCS for row in pmc[name])
    for row, from_pmc in training_rows:
        code = row["target_code"]
        key = pair_key(row)
        identity = (row["source_code"], normalized(row["source_name"]))
        if (
            row["exclude"]
            or code not in coa
            or key in held_out_pairs
            or identity in held_out_accounts
        ):
            continue
        example = training.setdefault(
            key,
            {
                "source_code": row["source_code"],
                "source_name": row["source_name"],
                "target_code": code,
                "target_name": coa[code]["target_name"],
                "review_note": row["review_note"],
                "training_pmc_count": 0,
            },
        )
        example["training_pmc_count"] += from_pmc
    golden = [
        {
            **{
                field: row[field]
                for field in (
                    "source_code",
                    "source_name",
                    "normal_balance",
                    "ending_balance",
                    "exclude",
                )
            },
            "expected_target_code": None if row["exclude"] else row["target_code"],
            "expected_routing_status": (
                "EXCLUDED"
                if row["exclude"]
                else "NEEDS_REVIEW"
                if row.get("changed") or not row["target_code"]
                else None
            ),
        }
        for row in held_out
    ]
    return list(training.values()), golden, corrected


def document_frequency(terms, documents):
    corpus = [normalized(document) for document in documents if document]
    result = Counter()
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?:s|es)?(?!\w)")
        result[term] = sum(bool(pattern.search(document)) for document in corpus)
    return result


def mine_rules(artifacts, pmc, reviewed, corrected):
    descriptions = [row["source_name"] for group in pmc.values() for row in group]
    descriptions.extend(row["source_name"] for row in reviewed)
    for row in rows(artifacts / ENTITY_2, "Mapping Data", 2):
        descriptions.extend((clean(row[1]), clean(row[6])))
    candidates = ABBREVIATIONS | TEAM_ABBREVIATIONS
    abbreviations = {
        term: candidates[term]
        for term, count in document_frequency(candidates, descriptions).most_common()
        if count >= (1 if term in TEAM_ABBREVIATIONS else 2)
    }
    # Reused comment templates provide evidence once for frequency ranking.
    review_text = list(dict.fromkeys(row["review_note"] for row in reviewed))
    review_text.extend(row["source_name"] for row in corrected)
    red_flags = [
        term
        for term, count in document_frequency(TRAP_TERMS, review_text).most_common()
        if count
    ]
    return abbreviations, red_flags


def prepare(raw_dir, extracted_dir, output_dir):
    artifacts = resolve_artifacts(raw_dir, extracted_dir)
    mapping_file = raw_dir / MAPPING_FILE
    coa = build_target_coa(coa_rows(artifacts, mapping_file))
    if not coa:
        raise ValueError("No postable accounts found in the target COA.")
    pmc = {name: list(pmc_rows(artifacts, name)) for name in TRAIN_PMCS + TEST_PMCS}
    reviewed = list(reviewed_rows(mapping_file))
    training, golden, corrected = build_examples(pmc, reviewed, coa)
    abbreviations, red_flags = mine_rules(artifacts, pmc, reviewed, corrected)
    outputs = {
        "target_coa.jsonl": list(coa.values()),
        "few_shot_kb.jsonl": training,
        "golden_test_set.jsonl": golden,
        "abbrev_map.json": abbreviations,
        "red_flag_keywords.json": red_flags,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temporary:
        for name, data in outputs.items():
            with (Path(temporary) / name).open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                if name.endswith(".jsonl"):
                    for record in data:
                        stream.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                else:
                    json.dump(
                        data, stream, indent=2, ensure_ascii=False, allow_nan=False
                    )
                    stream.write("\n")
        for name, data in outputs.items():
            (Path(temporary) / name).replace(output_dir / name)
            print(f"{name}: {len(data)} entries")
    print(f"Mapping Final: {len(corrected)} corrected rows, 100 sampled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("Given-Resources"))
    parser.add_argument(
        "--extracted-dir", type=Path, default=Path("data/raw_extracted")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    try:
        prepare(args.raw_dir, args.extracted_dir, args.output_dir)
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")


if __name__ == "__main__":
    main()
