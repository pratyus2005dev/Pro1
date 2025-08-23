from __future__ import annotations
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlparse
from .utils import tokens, jaccard, seq_ratio, coarse_type, string_value_jaccard, numeric_stat_similarity, date_range_overlap

class EnhancedFeaturizer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        self.name_embeddings = None
        
    def extract_metadata_features(self, ddl: Dict[str, Dict[str, str]], table_name: str, column_name: str) -> Dict[str, Any]:
        """Extract metadata features from DDL"""
        column_info = ddl.get(table_name, {}).get(column_name, {})
        
        features = {
            'is_primary_key': 0,
            'is_foreign_key': 0,
            'is_unique': 0,
            'is_nullable': 1,
            'has_default': 0,
            'data_type': 'unknown',
            'precision': 0,
            'scale': 0,
            'constraint_count': 0
        }
        
        if isinstance(column_info, dict):
            features.update({
                'is_primary_key': 1 if column_info.get('is_primary_key', False) else 0,
                'is_foreign_key': 1 if column_info.get('is_foreign_key', False) else 0,
                'is_unique': 1 if column_info.get('is_unique', False) else 0,
                'is_nullable': 0 if column_info.get('not_null', False) else 1,
                'has_default': 1 if column_info.get('default', None) is not None else 0,
                'data_type': column_info.get('type', 'unknown'),
                'precision': column_info.get('precision', 0),
                'scale': column_info.get('scale', 0),
                'constraint_count': len(column_info.get('constraints', []))
            })
        
        return features
    
    def advanced_fuzzy_scores(self, name1: str, name2: str) -> Dict[str, float]:
        """Calculate multiple fuzzy matching scores"""
        clean_name1 = re.sub(r'[^a-zA-Z0-9]', ' ', name1.lower()).strip()
        clean_name2 = re.sub(r'[^a-zA-Z0-9]', ' ', name2.lower()).strip()
        
        return {
            'levenshtein_ratio': fuzz.ratio(clean_name1, clean_name2) / 100.0,
            'partial_ratio': fuzz.partial_ratio(clean_name1, clean_name2) / 100.0,
            'token_sort_ratio': fuzz.token_sort_ratio(clean_name1, clean_name2) / 100.0,
            'token_set_ratio': fuzz.token_set_ratio(clean_name1, clean_name2) / 100.0,
            'jaro_winkler': fuzz.jaro_winkler(clean_name1, clean_name2),
            'sequence_ratio': seq_ratio(name1, name2),
            'token_jaccard': jaccard(tokens(name1), tokens(name2))
        }
    
    def semantic_similarity(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity using TF-IDF"""
        try:
            if not hasattr(self, '_tfidf_matrix'):
                # Initialize TF-IDF with sample names
                sample_names = [name1, name2, "id", "name", "date", "amount", "status"]
                self._tfidf_matrix = self.tfidf_vectorizer.fit_transform(sample_names)
            
            # Transform the two names
            names_vector = self.tfidf_vectorizer.transform([name1, name2])
            similarity = cosine_similarity(names_vector[0:1], names_vector[1:2])[0][0]
            return float(similarity)
        except:
            return 0.5
    
    def data_profile_features(self, series1: pd.Series, series2: pd.Series, dtype1: str, dtype2: str) -> Dict[str, float]:
        """Enhanced data profiling features"""
        features = {
            'null_rate_diff': abs(series1.isna().mean() - series2.isna().mean()),
            'cardinality_ratio': 0.0,
            'value_overlap': 0.0,
            'length_similarity': 0.0,
            'pattern_similarity': 0.0
        }
        
        # Cardinality ratio
        card1 = series1.nunique()
        card2 = series2.nunique()
        if card1 > 0 and card2 > 0:
            features['cardinality_ratio'] = min(card1, card2) / max(card1, card2)
        
        # Value overlap for string columns
        if coarse_type(dtype1) == "string" and coarse_type(dtype2) == "string":
            sample_size = min(1000, len(series1), len(series2))
            values1 = set(series1.dropna().astype(str).str.lower().head(sample_size))
            values2 = set(series2.dropna().astype(str).str.lower().head(sample_size))
            
            if values1 and values2:
                features['value_overlap'] = len(values1 & values2) / len(values1 | values2)
            
            # Length similarity
            len1 = series1.dropna().astype(str).str.len()
            len2 = series2.dropna().astype(str).str.len()
            if len(len1) > 0 and len(len2) > 0:
                features['length_similarity'] = 1.0 - abs(len1.median() - len2.median()) / max(len1.median(), len2.median(), 1)
        
        return features
    
    def column_features(
        self, 
        s_table: str, s_col: str, s_series: pd.Series, s_dtype: str,
        t_table: str, t_col: str, t_series: pd.Series, t_dtype: str,
        ddl: Dict[str, Dict[str, str]]
    ) -> Dict[str, float]:
        """Enhanced column features combining all similarity measures"""
        
        # Basic type matching
        s_type = coarse_type(s_dtype)
        t_type = coarse_type(t_dtype)
        
        # Fuzzy name matching
        fuzzy_scores = self.advanced_fuzzy_scores(s_col, t_col)
        
        # Semantic similarity
        semantic_sim = self.semantic_similarity(s_col, t_col)
        
        # Metadata features
        s_meta = self.extract_metadata_features(ddl, s_table, s_col)
        t_meta = self.extract_metadata_features(ddl, t_table, t_col)
        
        # Data profiling
        profile_features = self.data_profile_features(s_series, t_series, s_dtype, t_dtype)
        
        # Type-specific features
        type_features = {
            'coarse_type_match': 1.0 if s_type == t_type else 0.0,
            'strlen_med_diff': 0.0,
            'string_val_jaccard': 0.0,
            'numeric_stat_sim': 0.0,
            'date_overlap': 0.0,
        }
        
        if s_type == "string" and t_type == "string":
            s_len = s_series.dropna().astype(str).str.len()
            t_len = t_series.dropna().astype(str).str.len()
            if len(s_len) > 0 and len(t_len) > 0:
                type_features["strlen_med_diff"] = float(abs(s_len.median() - t_len.median()) / (max(s_len.median(), t_len.median(), 1)))
            type_features["string_val_jaccard"] = string_value_jaccard(s_series, t_series)
        elif s_type == "numeric" and t_type == "numeric":
            type_features["numeric_stat_sim"] = numeric_stat_similarity(s_series, t_series)
        elif s_type == "date" and t_type == "date":
            type_features["date_overlap"] = date_range_overlap(s_series, t_series)
        
        # Metadata similarity
        metadata_similarity = {
            'pk_similarity': 1.0 if s_meta['is_primary_key'] == t_meta['is_primary_key'] else 0.0,
            'fk_similarity': 1.0 if s_meta['is_foreign_key'] == t_meta['is_foreign_key'] else 0.0,
            'unique_similarity': 1.0 if s_meta['is_unique'] == t_meta['is_unique'] else 0.0,
            'nullable_similarity': 1.0 if s_meta['is_nullable'] == t_meta['is_nullable'] else 0.0,
            'default_similarity': 1.0 if s_meta['has_default'] == t_meta['has_default'] else 0.0,
            'type_similarity': 1.0 if s_meta['data_type'] == t_meta['data_type'] else 0.0
        }
        
        # Combine all features
        all_features = {
            **fuzzy_scores,
            'semantic_similarity': semantic_sim,
            **type_features,
            **profile_features,
            **metadata_similarity
        }
        
        return all_features

def to_frame(feature_rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(feature_rows)