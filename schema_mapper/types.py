from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set


@dataclass
class ColumnMeta:
	name: str
	data_type: str
	length: Optional[int] = None
	nullable: Optional[bool] = None
	default: Optional[str] = None


@dataclass
class TableMeta:
	name: str
	columns: Dict[str, ColumnMeta] = field(default_factory=dict)


DDLSchema = Dict[str, TableMeta]


@dataclass
class ColumnProfile:
	table_name: str
	column_name: str
	inferred_dtype: str
	is_numeric: bool
	is_string: bool
	sample_count: int
	null_fraction: float
	num_unique: int
	unique_fraction: float
	length_mean: Optional[float]
	length_std: Optional[float]
	numeric_mean: Optional[float]
	numeric_std: Optional[float]
	numeric_min: Optional[float]
	numeric_max: Optional[float]
	top_values: List[Tuple[str, int]]
	name_tokens: Set[str]


@dataclass
class TableProfile:
	table_name: str
	columns: Dict[str, ColumnProfile] = field(default_factory=dict)


@dataclass
class DatasetProfiles:
	source: Dict[str, TableProfile]
	target: Dict[str, TableProfile]
	ddl: DDLSchema


PairFeatures = Dict[str, float]