import unicodedata


def code_key(value):
    if value is None:
        return None
    code = unicodedata.normalize("NFKC", str(value))
    return "".join(char for char in code.upper() if char.isalnum()) or None


def hierarchy_text(name: str, parent_path: str) -> str:
    parents = [parent.strip() for parent in parent_path.split(" > ") if parent.strip()]
    if not parents:
        return name
    if len(parents) == 1:
        return f"{name} — {parents[0]}"
    return f"{name} — {parents[-1]} ({parents[0]})"
