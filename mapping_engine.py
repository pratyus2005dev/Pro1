from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from dataclasses import dataclass
from enum import Enum
import json

class MappingType(Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"

@dataclass
class MappingScore:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    ml_score: float
    fuzzy_score: float
    combined_score: float
    mapping_type: MappingType
    confidence: float
    metadata: Dict[str, Any]

class MappingEngine:
    """Advanced mapping engine supporting various mapping types"""
    
    def __init__(self, featurizer, model, ddl: Dict[str, Dict[str, str]]):
        self.featurizer = featurizer
        self.model = model
        self.ddl = ddl
        self.mapping_scores = []
        
    def calculate_fuzzy_score(self, source_col: str, target_col: str) -> float:
        """Calculate comprehensive fuzzy matching score"""
        fuzzy_scores = self.featurizer.advanced_fuzzy_scores(source_col, target_col)
        
        # Weighted combination of fuzzy scores
        weights = {
            'levenshtein_ratio': 0.25,
            'partial_ratio': 0.20,
            'token_sort_ratio': 0.20,
            'token_set_ratio': 0.15,
            'jaro_winkler': 0.10,
            'sequence_ratio': 0.05,
            'token_jaccard': 0.05
        }
        
        weighted_score = sum(
            fuzzy_scores[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return weighted_score
    
    def generate_mapping_candidates(
        self, 
        source_tables: Dict[str, pd.DataFrame],
        target_tables: Dict[str, pd.DataFrame],
        table_pairs: List[List[str]]
    ) -> List[MappingScore]:
        """Generate all possible mapping candidates with scores"""
        
        candidates = []
        
        for s_table, t_table in table_pairs:
            if s_table not in source_tables or t_table not in target_tables:
                continue
                
            s_df = source_tables[s_table]
            t_df = target_tables[t_table]
            
            for s_col in s_df.columns:
                for t_col in t_df.columns:
                    # Calculate features
                    features = self.featurizer.column_features(
                        s_table, s_col, s_df[s_col], 
                        self.ddl.get(s_table, {}).get(s_col, "string"),
                        t_table, t_col, t_df[t_col], 
                        self.ddl.get(t_table, {}).get(t_col, "string"),
                        self.ddl
                    )
                    
                    # Get ML score
                    feature_vector = np.array([list(features.values())])
                    ml_score = float(self.model.predict_proba(feature_vector)[0])
                    
                    # Get fuzzy score
                    fuzzy_score = self.calculate_fuzzy_score(s_col, t_col)
                    
                    # Combined score (weighted average)
                    combined_score = 0.7 * ml_score + 0.3 * fuzzy_score
                    
                    # Determine mapping type based on metadata
                    mapping_type = self._determine_mapping_type(s_table, s_col, t_table, t_col)
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(ml_score, fuzzy_score, features)
                    
                    candidate = MappingScore(
                        source_table=s_table,
                        source_column=s_col,
                        target_table=t_table,
                        target_column=t_col,
                        ml_score=ml_score,
                        fuzzy_score=fuzzy_score,
                        combined_score=combined_score,
                        mapping_type=mapping_type,
                        confidence=confidence,
                        metadata=features
                    )
                    
                    candidates.append(candidate)
        
        return candidates
    
    def _determine_mapping_type(self, s_table: str, s_col: str, t_table: str, t_col: str) -> MappingType:
        """Determine the mapping type based on metadata"""
        s_meta = self.featurizer.extract_metadata_features(self.ddl, s_table, s_col)
        t_meta = self.featurizer.extract_metadata_features(self.ddl, t_table, t_col)
        
        # Check for primary key relationships
        if s_meta['is_primary_key'] and t_meta['is_primary_key']:
            return MappingType.ONE_TO_ONE
        
        # Check for foreign key relationships
        if s_meta['is_foreign_key'] or t_meta['is_foreign_key']:
            return MappingType.MANY_TO_ONE
        
        # Default to one-to-one for now
        return MappingType.ONE_TO_ONE
    
    def _calculate_confidence(self, ml_score: float, fuzzy_score: float, features: Dict[str, float]) -> float:
        """Calculate confidence score based on multiple factors"""
        # Base confidence from scores
        base_confidence = (ml_score + fuzzy_score) / 2
        
        # Boost confidence for high-quality matches
        if features.get('coarse_type_match', 0) == 1.0:
            base_confidence *= 1.1
        
        if features.get('pk_similarity', 0) == 1.0:
            base_confidence *= 1.2
        
        if features.get('semantic_similarity', 0) > 0.8:
            base_confidence *= 1.1
        
        return min(base_confidence, 1.0)
    
    def optimize_one_to_one_mappings(
        self, 
        candidates: List[MappingScore], 
        threshold: float = 0.5
    ) -> List[MappingScore]:
        """Optimize one-to-one mappings using Hungarian algorithm"""
        
        # Filter candidates by threshold and mapping type
        valid_candidates = [
            c for c in candidates 
            if c.combined_score >= threshold and c.mapping_type == MappingType.ONE_TO_ONE
        ]
        
        if not valid_candidates:
            return []
        
        # Group by table pairs
        table_pairs = {}
        for candidate in valid_candidates:
            pair_key = (candidate.source_table, candidate.target_table)
            if pair_key not in table_pairs:
                table_pairs[pair_key] = []
            table_pairs[pair_key].append(candidate)
        
        optimized_mappings = []
        
        for (s_table, t_table), pair_candidates in table_pairs.items():
            # Create cost matrix for Hungarian algorithm
            source_cols = list(set(c.source_column for c in pair_candidates))
            target_cols = list(set(c.target_column for c in pair_candidates))
            
            cost_matrix = np.zeros((len(source_cols), len(target_cols)))
            
            for i, s_col in enumerate(source_cols):
                for j, t_col in enumerate(target_cols):
                    # Find the candidate for this pair
                    candidate = next(
                        (c for c in pair_candidates 
                         if c.source_column == s_col and c.target_column == t_col),
                        None
                    )
                    
                    if candidate:
                        # Use negative score as cost (Hungarian minimizes cost)
                        cost_matrix[i, j] = -candidate.combined_score
                    else:
                        cost_matrix[i, j] = 0  # No match possible
            
            # Apply Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Extract optimal mappings
            for row_idx, col_idx in zip(row_indices, col_indices):
                s_col = source_cols[row_idx]
                t_col = target_cols[col_idx]
                
                candidate = next(
                    (c for c in pair_candidates 
                     if c.source_column == s_col and c.target_column == t_col),
                    None
                )
                
                if candidate and candidate.combined_score >= threshold:
                    optimized_mappings.append(candidate)
        
        return optimized_mappings
    
    def optimize_one_to_many_mappings(
        self, 
        candidates: List[MappingScore], 
        threshold: float = 0.5,
        max_targets_per_source: int = 3
    ) -> List[MappingScore]:
        """Optimize one-to-many mappings"""
        
        valid_candidates = [
            c for c in candidates 
            if c.combined_score >= threshold and c.mapping_type == MappingType.ONE_TO_MANY
        ]
        
        if not valid_candidates:
            return []
        
        # Group by source columns
        source_groups = {}
        for candidate in valid_candidates:
            key = (candidate.source_table, candidate.source_column)
            if key not in source_groups:
                source_groups[key] = []
            source_groups[key].append(candidate)
        
        optimized_mappings = []
        
        for (s_table, s_col), group_candidates in source_groups.items():
            # Sort by score and take top-k
            sorted_candidates = sorted(
                group_candidates, 
                key=lambda x: x.combined_score, 
                reverse=True
            )
            
            # Take top candidates up to max_targets_per_source
            top_candidates = sorted_candidates[:max_targets_per_source]
            optimized_mappings.extend(top_candidates)
        
        return optimized_mappings
    
    def optimize_many_to_one_mappings(
        self, 
        candidates: List[MappingScore], 
        threshold: float = 0.5,
        max_sources_per_target: int = 3
    ) -> List[MappingScore]:
        """Optimize many-to-one mappings"""
        
        valid_candidates = [
            c for c in candidates 
            if c.combined_score >= threshold and c.mapping_type == MappingType.MANY_TO_ONE
        ]
        
        if not valid_candidates:
            return []
        
        # Group by target columns
        target_groups = {}
        for candidate in valid_candidates:
            key = (candidate.target_table, candidate.target_column)
            if key not in target_groups:
                target_groups[key] = []
            target_groups[key].append(candidate)
        
        optimized_mappings = []
        
        for (t_table, t_col), group_candidates in target_groups.items():
            # Sort by score and take top-k
            sorted_candidates = sorted(
                group_candidates, 
                key=lambda x: x.combined_score, 
                reverse=True
            )
            
            # Take top candidates up to max_sources_per_target
            top_candidates = sorted_candidates[:max_sources_per_target]
            optimized_mappings.extend(top_candidates)
        
        return optimized_mappings
    
    def generate_final_mappings(
        self,
        source_tables: Dict[str, pd.DataFrame],
        target_tables: Dict[str, pd.DataFrame],
        table_pairs: List[List[str]],
        threshold: float = 0.5
    ) -> Dict[str, List[MappingScore]]:
        """Generate final optimized mappings for all types"""
        
        # Generate all candidates
        candidates = self.generate_mapping_candidates(
            source_tables, target_tables, table_pairs
        )
        
        # Optimize each mapping type
        one_to_one = self.optimize_one_to_one_mappings(candidates, threshold)
        one_to_many = self.optimize_one_to_many_mappings(candidates, threshold)
        many_to_one = self.optimize_many_to_one_mappings(candidates, threshold)
        
        # Many-to-many can be combinations of the above
        many_to_many = [
            c for c in candidates 
            if c.mapping_type == MappingType.MANY_TO_MANY and c.combined_score >= threshold
        ]
        
        return {
            'one_to_one': one_to_one,
            'one_to_many': one_to_many,
            'many_to_one': many_to_one,
            'many_to_many': many_to_many,
            'all_candidates': candidates
        }
    
    def export_to_csv(self, mappings: Dict[str, List[MappingScore]], output_path: str):
        """Export mappings to CSV format"""
        
        rows = []
        
        for mapping_type, mapping_list in mappings.items():
            if mapping_type == 'all_candidates':
                continue
                
            for mapping in mapping_list:
                row = {
                    'mapping_type': mapping_type,
                    'source_table': mapping.source_table,
                    'source_column': mapping.source_column,
                    'target_table': mapping.target_table,
                    'target_column': mapping.target_column,
                    'ml_score': mapping.ml_score,
                    'fuzzy_score': mapping.fuzzy_score,
                    'combined_score': mapping.combined_score,
                    'confidence': mapping.confidence,
                    'levenshtein_ratio': mapping.metadata.get('levenshtein_ratio', 0),
                    'semantic_similarity': mapping.metadata.get('semantic_similarity', 0),
                    'type_match': mapping.metadata.get('coarse_type_match', 0),
                    'pk_similarity': mapping.metadata.get('pk_similarity', 0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('combined_score', ascending=False)
        df.to_csv(output_path, index=False)
        
        return df
    
    def export_to_json(self, mappings: Dict[str, List[MappingScore]], output_path: str):
        """Export mappings to JSON format with detailed information"""
        
        output_data = {}
        
        for mapping_type, mapping_list in mappings.items():
            if mapping_type == 'all_candidates':
                continue
                
            output_data[mapping_type] = []
            
            for mapping in mapping_list:
                mapping_dict = {
                    'source_table': mapping.source_table,
                    'source_column': mapping.source_column,
                    'target_table': mapping.target_table,
                    'target_column': mapping.target_column,
                    'scores': {
                        'ml_score': mapping.ml_score,
                        'fuzzy_score': mapping.fuzzy_score,
                        'combined_score': mapping.combined_score,
                        'confidence': mapping.confidence
                    },
                    'metadata': mapping.metadata
                }
                output_data[mapping_type].append(mapping_dict)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        return output_data