import re
from typing import Dict, List, Optional

from .types import TableMeta, ColumnMeta, DDLSchema
from .utils import normalize_identifier


_CREATE_TABLE_RE = re.compile(
	r"create\s+table\s+(?:if\s+not\s+exists\s+)?([`\"\[]?[\w\.]+[`\"\]]?)\s*\((.*?)\)\s*;",
	re.IGNORECASE | re.DOTALL,
)

_COLUMN_LINE_RE = re.compile(
	r"^\s*([`\"\[]?[\w]+[`\"\]]?)\s+([\w]+)(?:\s*\((\d+)\))?(.*)$",
	re.IGNORECASE,
)


def _parse_column_line(line: str) -> Optional[ColumnMeta]:
	line_no_trailing = line.rstrip().rstrip(",")
	# Skip constraint lines
	upper = line_no_trailing.strip().upper()
	if upper.startswith("CONSTRAINT") or upper.startswith("PRIMARY KEY") or upper.startswith("FOREIGN KEY") or upper.startswith("UNIQUE") or upper.startswith("INDEX"):
		return None
	m = _COLUMN_LINE_RE.match(line_no_trailing)
	if not m:
		return None
	name_raw, dtype_raw, length_raw, tail = m.groups()
	name = normalize_identifier(name_raw)
	dtype = normalize_identifier(dtype_raw)
	length = int(length_raw) if length_raw else None
	nullable: Optional[bool] = None
	if "NOT NULL" in upper:
		nullable = False
	elif re.search(r"\bNULL\b", upper) is not None:
		nullable = True
	default = None
	m_def = re.search(r"DEFAULT\s+([^\s,]+)", tail, flags=re.IGNORECASE)
	if m_def:
		default = m_def.group(1)
	return ColumnMeta(name=name, data_type=dtype, length=length, nullable=nullable, default=default)


def parse_ddl_files(paths: List[str]) -> DDLSchema:
	schema: DDLSchema = {}
	ddl_combined = ""
	for p in paths:
		with open(p, "r", encoding="utf-8", errors="ignore") as f:
			ddl_combined += f.read() + "\n"
	for m in _CREATE_TABLE_RE.finditer(ddl_combined):
		table_name_raw = m.group(1)
		columns_block = m.group(2)
		table_name = normalize_identifier(table_name_raw)
		table_meta = TableMeta(name=table_name)
		lines = [l for l in columns_block.split("\n") if l.strip()]
		for line in lines:
			col = _parse_column_line(line)
			if col is None:
				continue
			table_meta.columns[col.name] = col
		schema[table_name] = table_meta
	return schema