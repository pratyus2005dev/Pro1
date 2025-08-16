import re
from typing import Dict, List, Tuple

CREATE_RE = re.compile(r"CREATE\s+TABLE\s+([A-Za-z0-9_]+)\s*\((.*?)\);", re.S | re.I)

def parse_columns(block: str) -> List[Tuple[str, str]]:
    cols = []
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("--"):
            continue
        # crude: stop at constraints lines
        if re.search(r"\bPRIMARY\s+KEY\b|\bFOREIGN\s+KEY\b", line, re.I):
            continue
        m = re.match(r"([A-Za-z0-9_]+)\s+([A-Za-z0-9_()]+)", line)
        if m:
            cols.append((m.group(1), m.group(2)))
    return cols

def parse_ddl(sql_text: str) -> Dict[str, Dict[str, str]]:
    """
    Returns: {table: {column: sql_type}}
    """
    tables: Dict[str, Dict[str, str]] = {}
    for m in CREATE_RE.finditer(sql_text):
        table = m.group(1)
        cols_block = m.group(2)
        cols = parse_columns(cols_block)
        tables[table] = {c: t for c, t in cols}
    return tables

def load_ddl(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_ddl(text)
