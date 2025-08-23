#!/usr/bin/env python3
"""
Schema Mapping Automation Tool

This tool performs automated database schema mapping between source and target databases
using machine learning models and fuzzy matching algorithms. It supports various mapping
types (1:1, 1:many, many:1, many:many) and generates confidence scores for each mapping.
"""

import pandas as pd
import numpy as np
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML and Data Science imports
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# String similarity imports
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import Levenshtein

# SQL parsing
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Name


@dataclass
class ColumnInfo:
    """Represents column metadata"""
    name: str
    data_type: str
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    default_value: Optional[str] = None
    constraints: List[str] = None
    table_name: str = ""
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


@dataclass
class TableInfo:
    """Represents table metadata"""
    name: str
    columns: List[ColumnInfo]
    primary_keys: List[str] = None
    foreign_keys: List[str] = None
    indexes: List[str] = None
    
    def __post_init__(self):
        if self.primary_keys is None:
            self.primary_keys = []
        if self.foreign_keys is None:
            self.foreign_keys = []
        if self.indexes is None:
            self.indexes = []


@dataclass
class MappingResult:
    """Represents a mapping result between source and target columns"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    fuzzy_score: float
    ml_score: float
    combined_score: float
    mapping_type: str  # '1:1', '1:many', 'many:1', 'many:many'
    confidence: str  # 'high', 'medium', 'low'
    metadata_features: Dict
    data_features: Dict


class DDLParser:
    """Parses DDL SQL files to extract table and column metadata"""
    
    def __init__(self):
        self.tables = {}
        
    def parse_ddl_file(self, ddl_path: str) -> Dict[str, TableInfo]:
        """Parse DDL file and extract table information"""
        with open(ddl_path, 'r', encoding='utf-8') as f:
            ddl_content = f.read()
        
        # Parse SQL statements
        statements = sqlparse.split(ddl_content)
        
        for statement in statements:
            if statement.strip():
                parsed = sqlparse.parse(statement)[0]
                self._extract_table_info(parsed)
        
        return self.tables
    
    def _extract_table_info(self, parsed_statement):
        """Extract table information from parsed SQL statement"""
        statement_type = self._get_statement_type(parsed_statement)
        
        if statement_type == 'CREATE TABLE':
            self._parse_create_table(parsed_statement)
    
    def _get_statement_type(self, parsed_statement):
        """Determine the type of SQL statement"""
        statement_str = str(parsed_statement).strip().upper()
        if 'CREATE TABLE' in statement_str:
            return 'CREATE TABLE'
        return 'UNKNOWN'
    
    def _parse_create_table(self, parsed_statement):
        """Parse CREATE TABLE statement"""
        statement_str = str(parsed_statement)
        
        # Extract table name using regex
        import re
        table_match = re.search(r'CREATE\s+TABLE\s+([^\s\(]+)', statement_str, re.IGNORECASE)
        if not table_match:
            return
            
        table_name = table_match.group(1).strip('`"[]')
        
        # Parse column definitions
        columns = self._parse_column_definitions(parsed_statement, table_name)
        if columns:  # Only add if we found columns
            self.tables[table_name] = TableInfo(name=table_name, columns=columns)
    
    def _parse_column_definitions(self, parsed_statement, table_name):
        """Parse column definitions from CREATE TABLE statement"""
        columns = []
        statement_str = str(parsed_statement)
        
        # Simple regex-based parsing for column definitions
        # This is a simplified approach - in production, you'd want more robust parsing
        column_pattern = r'(\w+)\s+(\w+(?:\(\d+(?:,\d+)?\))?)\s*([^,\)]*)'
        
        # Extract the part between parentheses
        paren_start = statement_str.find('(')
        paren_end = statement_str.rfind(')')
        
        if paren_start != -1 and paren_end != -1:
            columns_section = statement_str[paren_start+1:paren_end]
            
            # Split by comma, but be careful with nested parentheses
            column_defs = self._split_column_definitions(columns_section)
            
            for col_def in column_defs:
                col_def = col_def.strip()
                if col_def and not col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX', 'KEY')):
                    column = self._parse_single_column(col_def, table_name)
                    if column:
                        columns.append(column)
        
        return columns
    
    def _split_column_definitions(self, columns_section):
        """Split column definitions by comma, handling nested parentheses"""
        definitions = []
        current_def = ""
        paren_count = 0
        
        for char in columns_section:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                definitions.append(current_def)
                current_def = ""
                continue
            
            current_def += char
        
        if current_def.strip():
            definitions.append(current_def)
        
        return definitions
    
    def _parse_single_column(self, col_def, table_name):
        """Parse a single column definition"""
        parts = col_def.strip().split()
        if len(parts) < 2:
            return None
        
        column_name = parts[0].strip('`"[]')
        data_type_raw = parts[1]
        
        # Extract data type, length, precision, scale
        data_type, length, precision, scale = self._parse_data_type(data_type_raw)
        
        # Check for constraints
        nullable = 'NOT NULL' not in col_def.upper()
        constraints = []
        
        if 'PRIMARY KEY' in col_def.upper():
            constraints.append('PRIMARY KEY')
        if 'UNIQUE' in col_def.upper():
            constraints.append('UNIQUE')
        if 'AUTO_INCREMENT' in col_def.upper():
            constraints.append('AUTO_INCREMENT')
        
        # Extract default value
        default_value = None
        default_match = re.search(r'DEFAULT\s+([^,\s]+)', col_def.upper())
        if default_match:
            default_value = default_match.group(1)
        
        return ColumnInfo(
            name=column_name,
            data_type=data_type,
            length=length,
            precision=precision,
            scale=scale,
            nullable=nullable,
            default_value=default_value,
            constraints=constraints,
            table_name=table_name
        )
    
    def _parse_data_type(self, data_type_raw):
        """Parse data type string to extract type, length, precision, scale"""
        # Handle types like VARCHAR(255), DECIMAL(10,2), etc.
        match = re.match(r'(\w+)(?:\((\d+)(?:,(\d+))?\))?', data_type_raw)
        
        if match:
            data_type = match.group(1).upper()
            length = int(match.group(2)) if match.group(2) else None
            precision = int(match.group(2)) if match.group(2) else None
            scale = int(match.group(3)) if match.group(3) else None
            
            return data_type, length, precision, scale
        
        return data_type_raw.upper(), None, None, None


class FuzzyMatcher:
    """Handles fuzzy string matching for column and table names"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            lowercase=True
        )
    
    def calculate_fuzzy_score(self, source_name: str, target_name: str) -> float:
        """Calculate comprehensive fuzzy matching score"""
        scores = []
        
        # Levenshtein distance
        lev_score = 1 - (Levenshtein.distance(source_name.lower(), target_name.lower()) / 
                        max(len(source_name), len(target_name)))
        scores.append(lev_score)
        
        # Fuzzy ratio
        fuzzy_ratio = fuzz.ratio(source_name.lower(), target_name.lower()) / 100
        scores.append(fuzzy_ratio)
        
        # Fuzzy partial ratio
        fuzzy_partial = fuzz.partial_ratio(source_name.lower(), target_name.lower()) / 100
        scores.append(fuzzy_partial)
        
        # Token sort ratio
        token_sort = fuzz.token_sort_ratio(source_name.lower(), target_name.lower()) / 100
        scores.append(token_sort)
        
        # Token set ratio
        token_set = fuzz.token_set_ratio(source_name.lower(), target_name.lower()) / 100
        scores.append(token_set)
        
        # Sequence matcher
        seq_score = SequenceMatcher(None, source_name.lower(), target_name.lower()).ratio()
        scores.append(seq_score)
        
        # TF-IDF cosine similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([source_name.lower(), target_name.lower()])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores.append(cosine_sim)
        except:
            scores.append(0.0)
        
        # Weighted average (giving more weight to certain metrics)
        weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10]
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(1.0, max(0.0, weighted_score))
    
    def calculate_semantic_similarity(self, source_col: ColumnInfo, target_col: ColumnInfo) -> float:
        """Calculate semantic similarity based on column metadata"""
        score = 0.0
        
        # Data type similarity
        if source_col.data_type == target_col.data_type:
            score += 0.3
        elif self._are_compatible_types(source_col.data_type, target_col.data_type):
            score += 0.15
        
        # Length/precision similarity
        if source_col.length and target_col.length:
            length_diff = abs(source_col.length - target_col.length)
            max_length = max(source_col.length, target_col.length)
            length_similarity = 1 - (length_diff / max_length) if max_length > 0 else 1
            score += 0.2 * length_similarity
        
        # Nullable similarity
        if source_col.nullable == target_col.nullable:
            score += 0.1
        
        # Constraint similarity
        common_constraints = set(source_col.constraints) & set(target_col.constraints)
        total_constraints = set(source_col.constraints) | set(target_col.constraints)
        if total_constraints:
            constraint_similarity = len(common_constraints) / len(total_constraints)
            score += 0.2 * constraint_similarity
        
        # Name pattern similarity (e.g., both end with _id, both start with is_)
        name_pattern_score = self._calculate_name_pattern_similarity(source_col.name, target_col.name)
        score += 0.2 * name_pattern_score
        
        return min(1.0, score)
    
    def _are_compatible_types(self, type1: str, type2: str) -> bool:
        """Check if two data types are compatible"""
        # Define type compatibility groups
        numeric_types = {'INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE', 'REAL'}
        string_types = {'VARCHAR', 'CHAR', 'TEXT', 'LONGTEXT', 'MEDIUMTEXT', 'TINYTEXT', 'NVARCHAR', 'NCHAR'}
        date_types = {'DATE', 'DATETIME', 'TIMESTAMP', 'TIME'}
        binary_types = {'BLOB', 'LONGBLOB', 'MEDIUMBLOB', 'TINYBLOB', 'BINARY', 'VARBINARY'}
        
        type_groups = [numeric_types, string_types, date_types, binary_types]
        
        for group in type_groups:
            if type1 in group and type2 in group:
                return True
        
        return False
    
    def _calculate_name_pattern_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity based on naming patterns"""
        score = 0.0
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check for common prefixes
        common_prefixes = ['is_', 'has_', 'can_', 'should_', 'user_', 'customer_', 'order_', 'product_']
        for prefix in common_prefixes:
            if name1_lower.startswith(prefix) and name2_lower.startswith(prefix):
                score += 0.3
                break
        
        # Check for common suffixes
        common_suffixes = ['_id', '_key', '_code', '_name', '_date', '_time', '_flag', '_status', '_type']
        for suffix in common_suffixes:
            if name1_lower.endswith(suffix) and name2_lower.endswith(suffix):
                score += 0.3
                break
        
        # Check for common words in the middle
        name1_parts = re.split(r'[_\-\s]', name1_lower)
        name2_parts = re.split(r'[_\-\s]', name2_lower)
        
        common_parts = set(name1_parts) & set(name2_parts)
        if common_parts:
            score += 0.4 * (len(common_parts) / max(len(name1_parts), len(name2_parts)))
        
        return min(1.0, score)


class MLModelTrainer:
    """Trains and manages ML models for data-based similarity scoring"""
    
    def __init__(self):
        self.gradient_boosting_model = None
        self.random_forest_model = None
        self.xgboost_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def prepare_training_data(self, source_tables: Dict[str, TableInfo], 
                            target_tables: Dict[str, TableInfo],
                            reference_mappings: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        features = []
        labels = []
        
        # If reference mappings are provided, use them for supervised learning
        if reference_mappings:
            for mapping in reference_mappings:
                source_table = mapping.get('source_table')
                source_column = mapping.get('source_column')
                target_table = mapping.get('target_table')
                target_column = mapping.get('target_column')
                is_correct = mapping.get('is_correct', 1)
                
                if (source_table in source_tables and target_table in target_tables):
                    source_col = self._find_column(source_tables[source_table], source_column)
                    target_col = self._find_column(target_tables[target_table], target_column)
                    
                    if source_col and target_col:
                        feature_vector = self._extract_features(source_col, target_col)
                        features.append(feature_vector)
                        labels.append(is_correct)
        
        # Generate synthetic training data if no reference mappings
        if not features:
            features, labels = self._generate_synthetic_training_data(source_tables, target_tables)
        
        return np.array(features), np.array(labels)
    
    def _find_column(self, table: TableInfo, column_name: str) -> Optional[ColumnInfo]:
        """Find a column in a table by name"""
        for col in table.columns:
            if col.name.lower() == column_name.lower():
                return col
        return None
    
    def _extract_features(self, source_col: ColumnInfo, target_col: ColumnInfo) -> List[float]:
        """Extract feature vector for ML model"""
        features = []
        
        # Name similarity features
        fuzzy_matcher = FuzzyMatcher()
        name_similarity = fuzzy_matcher.calculate_fuzzy_score(source_col.name, target_col.name)
        features.append(name_similarity)
        
        # Data type features
        features.append(1.0 if source_col.data_type == target_col.data_type else 0.0)
        features.append(1.0 if fuzzy_matcher._are_compatible_types(source_col.data_type, target_col.data_type) else 0.0)
        
        # Length features
        if source_col.length and target_col.length:
            length_ratio = min(source_col.length, target_col.length) / max(source_col.length, target_col.length)
            features.append(length_ratio)
        else:
            features.append(0.5)  # neutral value when length is unknown
        
        # Nullable feature
        features.append(1.0 if source_col.nullable == target_col.nullable else 0.0)
        
        # Constraint features
        common_constraints = set(source_col.constraints) & set(target_col.constraints)
        total_constraints = set(source_col.constraints) | set(target_col.constraints)
        constraint_similarity = len(common_constraints) / len(total_constraints) if total_constraints else 0
        features.append(constraint_similarity)
        
        # Name length features
        name_len_ratio = min(len(source_col.name), len(target_col.name)) / max(len(source_col.name), len(target_col.name))
        features.append(name_len_ratio)
        
        # Pattern matching features
        pattern_similarity = fuzzy_matcher._calculate_name_pattern_similarity(source_col.name, target_col.name)
        features.append(pattern_similarity)
        
        # Table name similarity
        table_similarity = fuzzy_matcher.calculate_fuzzy_score(source_col.table_name, target_col.table_name)
        features.append(table_similarity)
        
        return features
    
    def _generate_synthetic_training_data(self, source_tables: Dict[str, TableInfo], 
                                        target_tables: Dict[str, TableInfo]) -> Tuple[List, List]:
        """Generate synthetic training data when no reference mappings are available"""
        features = []
        labels = []
        
        fuzzy_matcher = FuzzyMatcher()
        
        # Generate positive examples (likely matches)
        for source_table_name, source_table in source_tables.items():
            for source_col in source_table.columns:
                best_matches = []
                
                for target_table_name, target_table in target_tables.items():
                    for target_col in target_table.columns:
                        similarity = fuzzy_matcher.calculate_fuzzy_score(source_col.name, target_col.name)
                        if similarity > 0.7:  # High similarity threshold for positive examples
                            best_matches.append((target_col, similarity))
                
                # Sort by similarity and take top matches
                best_matches.sort(key=lambda x: x[1], reverse=True)
                
                for target_col, similarity in best_matches[:2]:  # Top 2 matches
                    feature_vector = self._extract_features(source_col, target_col)
                    features.append(feature_vector)
                    labels.append(1)  # Positive example
        
        # Generate negative examples (unlikely matches)
        import random
        random.seed(42)
        
        source_cols = []
        target_cols = []
        
        for table in source_tables.values():
            source_cols.extend(table.columns)
        
        for table in target_tables.values():
            target_cols.extend(table.columns)
        
        # Generate random negative pairs
        for _ in range(len(features)):  # Same number of negative as positive examples
            source_col = random.choice(source_cols)
            target_col = random.choice(target_cols)
            
            similarity = fuzzy_matcher.calculate_fuzzy_score(source_col.name, target_col.name)
            if similarity < 0.3:  # Low similarity threshold for negative examples
                feature_vector = self._extract_features(source_col, target_col)
                features.append(feature_vector)
                labels.append(0)  # Negative example
        
        return features, labels
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple ML models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting
        self.gradient_boosting_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.gradient_boosting_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest
        self.random_forest_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.random_forest_model.fit(X_train_scaled, y_train)
        
        # Train XGBoost
        self.xgboost_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.xgboost_model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Print training results
        gb_pred = self.gradient_boosting_model.predict(X_test_scaled)
        try:
            rf_pred = self.random_forest_model.predict_proba(X_test_scaled)[:, 1]
            rf_accuracy = accuracy_score(y_test, (rf_pred > 0.5).astype(int))
        except IndexError:
            # Handle case where there's only one class
            rf_pred = self.random_forest_model.predict(X_test_scaled)
            rf_accuracy = accuracy_score(y_test, rf_pred)
        xgb_pred = self.xgboost_model.predict(X_test_scaled)
        
        print(f"Gradient Boosting RMSE: {np.sqrt(np.mean((gb_pred - y_test) ** 2)):.4f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"XGBoost RMSE: {np.sqrt(np.mean((xgb_pred - y_test) ** 2)):.4f}")
    
    def predict_similarity(self, source_col: ColumnInfo, target_col: ColumnInfo) -> float:
        """Predict similarity score using trained models"""
        if not self.is_trained:
            return 0.5  # Default score if models not trained
        
        feature_vector = self._extract_features(source_col, target_col)
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get predictions from all models
        gb_pred = self.gradient_boosting_model.predict(feature_vector_scaled)[0]
        try:
            rf_pred = self.random_forest_model.predict_proba(feature_vector_scaled)[0, 1]
        except IndexError:
            # Handle case where there's only one class
            rf_pred = self.random_forest_model.predict(feature_vector_scaled)[0]
        xgb_pred = self.xgboost_model.predict(feature_vector_scaled)[0]
        
        # Ensemble prediction (weighted average)
        ensemble_score = 0.4 * gb_pred + 0.3 * rf_pred + 0.3 * xgb_pred
        
        return max(0.0, min(1.0, ensemble_score))


class SchemaMappingEngine:
    """Main engine for schema mapping automation"""
    
    def __init__(self):
        self.ddl_parser = DDLParser()
        self.fuzzy_matcher = FuzzyMatcher()
        self.ml_trainer = MLModelTrainer()
        self.source_tables = {}
        self.target_tables = {}
        self.mappings = []
    
    def load_schemas(self, source_ddl_path: str, target_ddl_path: str):
        """Load source and target schemas from DDL files"""
        print("Loading source schema...")
        self.source_tables = self.ddl_parser.parse_ddl_file(source_ddl_path)
        
        print("Loading target schema...")
        self.ddl_parser.tables = {}  # Reset parser
        self.target_tables = self.ddl_parser.parse_ddl_file(target_ddl_path)
        
        print(f"Loaded {len(self.source_tables)} source tables and {len(self.target_tables)} target tables")
    
    def train_models(self, reference_mappings_path: Optional[str] = None):
        """Train ML models for similarity prediction"""
        print("Preparing training data...")
        
        reference_mappings = None
        if reference_mappings_path and Path(reference_mappings_path).exists():
            with open(reference_mappings_path, 'r') as f:
                reference_mappings = json.load(f)
        
        X, y = self.ml_trainer.prepare_training_data(
            self.source_tables, 
            self.target_tables, 
            reference_mappings
        )
        
        print("Training ML models...")
        self.ml_trainer.train_models(X, y)
        print("Model training completed!")
    
    def generate_mappings(self) -> List[MappingResult]:
        """Generate all possible mappings with scores"""
        print("Generating mappings...")
        mappings = []
        
        for source_table_name, source_table in self.source_tables.items():
            for source_col in source_table.columns:
                candidates = []
                
                # Find all potential target candidates
                for target_table_name, target_table in self.target_tables.items():
                    for target_col in target_table.columns:
                        # Calculate fuzzy score
                        fuzzy_score = self.fuzzy_matcher.calculate_fuzzy_score(
                            source_col.name, target_col.name
                        )
                        
                        # Calculate ML score
                        ml_score = self.ml_trainer.predict_similarity(source_col, target_col)
                        
                        # Calculate combined score
                        combined_score = 0.6 * fuzzy_score + 0.4 * ml_score
                        
                        # Extract features for metadata
                        metadata_features = {
                            'source_data_type': source_col.data_type,
                            'target_data_type': target_col.data_type,
                            'type_match': source_col.data_type == target_col.data_type,
                            'source_nullable': source_col.nullable,
                            'target_nullable': target_col.nullable,
                            'source_constraints': source_col.constraints,
                            'target_constraints': target_col.constraints
                        }
                        
                        data_features = {
                            'fuzzy_name_score': fuzzy_score,
                            'ml_similarity_score': ml_score,
                            'semantic_similarity': self.fuzzy_matcher.calculate_semantic_similarity(source_col, target_col)
                        }
                        
                        candidates.append({
                            'target_table': target_table_name,
                            'target_column': target_col.name,
                            'fuzzy_score': fuzzy_score,
                            'ml_score': ml_score,
                            'combined_score': combined_score,
                            'metadata_features': metadata_features,
                            'data_features': data_features
                        })
                
                # Sort candidates by combined score
                candidates.sort(key=lambda x: x['combined_score'], reverse=True)
                
                # Create mappings based on scores and determine mapping types
                for i, candidate in enumerate(candidates[:5]):  # Top 5 candidates
                    confidence = self._determine_confidence(candidate['combined_score'])
                    mapping_type = self._determine_mapping_type(
                        source_table_name, source_col.name,
                        candidate['target_table'], candidate['target_column'],
                        candidates
                    )
                    
                    mapping = MappingResult(
                        source_table=source_table_name,
                        source_column=source_col.name,
                        target_table=candidate['target_table'],
                        target_column=candidate['target_column'],
                        fuzzy_score=candidate['fuzzy_score'],
                        ml_score=candidate['ml_score'],
                        combined_score=candidate['combined_score'],
                        mapping_type=mapping_type,
                        confidence=confidence,
                        metadata_features=candidate['metadata_features'],
                        data_features=candidate['data_features']
                    )
                    
                    mappings.append(mapping)
        
        self.mappings = mappings
        return mappings
    
    def _determine_confidence(self, combined_score: float) -> str:
        """Determine confidence level based on combined score"""
        if combined_score >= 0.8:
            return 'high'
        elif combined_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _determine_mapping_type(self, source_table: str, source_column: str,
                              target_table: str, target_column: str,
                              all_candidates: List[Dict]) -> str:
        """Determine mapping type (1:1, 1:many, many:1, many:many)"""
        # Count how many high-scoring targets this source maps to
        high_score_targets = [c for c in all_candidates if c['combined_score'] >= 0.7]
        
        # Count how many sources map to this target (simplified heuristic)
        # In a full implementation, you'd track this across all mappings
        
        if len(high_score_targets) == 1:
            return '1:1'
        elif len(high_score_targets) > 1:
            return '1:many'
        else:
            # For now, default to 1:1 for lower scores
            # In practice, you'd analyze the full mapping matrix
            return '1:1'
    
    def export_to_csv(self, output_path: str, top_n: int = 1):
        """Export mappings to CSV file"""
        print(f"Exporting mappings to {output_path}...")
        
        # Group by source column and take top N mappings
        grouped_mappings = defaultdict(list)
        for mapping in self.mappings:
            key = f"{mapping.source_table}.{mapping.source_column}"
            grouped_mappings[key].append(mapping)
        
        # Sort each group and take top N
        export_data = []
        for source_key, mappings in grouped_mappings.items():
            mappings.sort(key=lambda x: x.combined_score, reverse=True)
            for mapping in mappings[:top_n]:
                export_data.append({
                    'source_table': mapping.source_table,
                    'source_column': mapping.source_column,
                    'target_table': mapping.target_table,
                    'target_column': mapping.target_column,
                    'fuzzy_score': round(mapping.fuzzy_score, 4),
                    'ml_score': round(mapping.ml_score, 4),
                    'combined_score': round(mapping.combined_score, 4),
                    'mapping_type': mapping.mapping_type,
                    'confidence': mapping.confidence,
                    'source_data_type': mapping.metadata_features.get('source_data_type'),
                    'target_data_type': mapping.metadata_features.get('target_data_type'),
                    'type_match': mapping.metadata_features.get('type_match'),
                    'semantic_similarity': round(mapping.data_features.get('semantic_similarity', 0), 4)
                })
        
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(export_data)} mappings to CSV")
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.mappings:
            print("No mappings generated yet.")
            return
        
        total_mappings = len(self.mappings)
        high_conf = len([m for m in self.mappings if m.confidence == 'high'])
        medium_conf = len([m for m in self.mappings if m.confidence == 'medium'])
        low_conf = len([m for m in self.mappings if m.confidence == 'low'])
        
        avg_fuzzy_score = np.mean([m.fuzzy_score for m in self.mappings])
        avg_ml_score = np.mean([m.ml_score for m in self.mappings])
        avg_combined_score = np.mean([m.combined_score for m in self.mappings])
        
        print("\n=== MAPPING SUMMARY ===")
        print(f"Total mappings: {total_mappings}")
        print(f"High confidence: {high_conf} ({high_conf/total_mappings*100:.1f}%)")
        print(f"Medium confidence: {medium_conf} ({medium_conf/total_mappings*100:.1f}%)")
        print(f"Low confidence: {low_conf} ({low_conf/total_mappings*100:.1f}%)")
        print(f"Average fuzzy score: {avg_fuzzy_score:.4f}")
        print(f"Average ML score: {avg_ml_score:.4f}")
        print(f"Average combined score: {avg_combined_score:.4f}")


def main():
    """Main function to run the schema mapping tool"""
    parser = argparse.ArgumentParser(description='Database Schema Mapping Automation Tool')
    parser.add_argument('--source-ddl', required=True, help='Path to source DDL file')
    parser.add_argument('--target-ddl', required=True, help='Path to target DDL file')
    parser.add_argument('--output', default='mappings.csv', help='Output CSV file path')
    parser.add_argument('--reference-mappings', help='Path to reference mappings JSON file (optional)')
    parser.add_argument('--top-n', type=int, default=3, help='Number of top mappings per source column')
    
    args = parser.parse_args()
    
    # Initialize the mapping engine
    engine = SchemaMappingEngine()
    
    try:
        # Load schemas
        engine.load_schemas(args.source_ddl, args.target_ddl)
        
        # Train models
        engine.train_models(args.reference_mappings)
        
        # Generate mappings
        mappings = engine.generate_mappings()
        
        # Export results
        engine.export_to_csv(args.output, args.top_n)
        
        # Print summary
        engine.print_summary()
        
        print(f"\n✅ Schema mapping completed successfully!")
        print(f"📊 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()