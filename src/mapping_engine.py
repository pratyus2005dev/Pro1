from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from scipy.optimize import linear_sum_assignment
from .featurizer import column_features, fuzzy_name_scores
from .model import ColumnMappingModel

class MappingType(Enum):
    ONE_ONE = "one_one"
    ONE_MANY = "one_many"
    MANY_ONE = "many_one"
    MANY_MANY = "many_many"

@dataclass
class MappingScore:
    """Represents a mapping score between source and target columns."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    ml_score: float
    fuzzy_score: float
    combined_score: float
    mapping_type: MappingType
    confidence: float
    features: Dict[str, float]

@dataclass
class MappingResult:
    """Represents the final mapping result."""
    source_table: str
    source_column: str
    target_table: str
    target_columns: List[str]
    mapping_type: MappingType
    scores: List[float]
    confidence: float
    alternatives: List[MappingScore]

class ColumnMappingEngine:
    """Advanced column mapping engine supporting various mapping types."""
    
    def __init__(self, model: ColumnMappingModel, fuzzy_weight: float = 0.3, ml_weight: float = 0.7):
        self.model = model
        self.fuzzy_weight = fuzzy_weight
        self.ml_weight = ml_weight
        self.mapping_scores: List[MappingScore] = []
        
    def calculate_all_scores(
        self,
        source_tables: Dict[str, pd.DataFrame],
        target_tables: Dict[str, pd.DataFrame],
        ddl: Dict[str, Dict[str, str]],
        table_pairs: List[List[str]]
    ) -> List[MappingScore]:
        """Calculate mapping scores for all possible column pairs."""
        scores = []
        
        for s_table, t_table in table_pairs:
            if s_table not in source_tables or t_table not in target_tables:
                continue
                
            for s_col in source_tables[s_table].columns:
                for t_col in target_tables[t_table].columns:
                    # Calculate features
                    s_series = source_tables[s_table][s_col]
                    t_series = target_tables[t_table][t_col]
                    s_dtype = ddl.get(s_table, {}).get(s_col, "string")
                    t_dtype = ddl.get(t_table, {}).get(t_col, "string")
                    
                    features = column_features(
                        s_table, s_col, s_series, s_dtype,
                        t_table, t_col, t_series, t_dtype
                    )
                    
                    # Calculate ML score
                    feature_vector = np.array([features[f] for f in self.model.feature_names_ if f in features])
                    ml_score = float(self.model.predict_proba([feature_vector])[0, 1])
                    
                    # Calculate fuzzy score
                    fuzzy_scores = fuzzy_name_scores(s_col, t_col)
                    fuzzy_score = np.mean([
                        fuzzy_scores["fuzzy_ratio"],
                        fuzzy_scores["fuzzy_token_sort_ratio"],
                        fuzzy_scores["rapidfuzz_ratio"]
                    ])
                    
                    # Combined score
                    combined_score = (self.fuzzy_weight * fuzzy_score + 
                                    self.ml_weight * ml_score)
                    
                    # Determine mapping type (simplified - will be refined later)
                    mapping_type = MappingType.ONE_ONE
                    
                    # Confidence based on score consistency
                    confidence = min(1.0, combined_score * 1.2)
                    
                    score = MappingScore(
                        source_table=s_table,
                        source_column=s_col,
                        target_table=t_table,
                        target_column=t_col,
                        ml_score=ml_score,
                        fuzzy_score=fuzzy_score,
                        combined_score=combined_score,
                        mapping_type=mapping_type,
                        confidence=confidence,
                        features=features
                    )
                    scores.append(score)
        
        self.mapping_scores = scores
        return scores
    
    def optimize_one_one_mappings(
        self,
        source_tables: Dict[str, pd.DataFrame],
        target_tables: Dict[str, pd.DataFrame],
        table_pairs: List[List[str]],
        threshold: float = 0.5
    ) -> List[MappingResult]:
        """Optimize one-to-one mappings using Hungarian algorithm."""
        results = []
        
        for s_table, t_table in table_pairs:
            if s_table not in source_tables or t_table not in target_tables:
                continue
            
            # Filter scores for this table pair
            table_scores = [s for s in self.mapping_scores 
                          if s.source_table == s_table and s.target_table == t_table
                          and s.combined_score >= threshold]
            
            if not table_scores:
                continue
            
            # Create cost matrix for Hungarian algorithm
            s_cols = list(set(s.source_column for s in table_scores))
            t_cols = list(set(s.target_column for s in table_scores))
            
            cost_matrix = np.zeros((len(s_cols), len(t_cols)))
            score_matrix = np.zeros((len(s_cols), len(t_cols)))
            
            for i, s_col in enumerate(s_cols):
                for j, t_col in enumerate(t_cols):
                    matching_score = next((s for s in table_scores 
                                         if s.source_column == s_col and s.target_column == t_col), None)
                    if matching_score:
                        score_matrix[i, j] = matching_score.combined_score
                        cost_matrix[i, j] = 1.0 - matching_score.combined_score  # Convert to cost
                    else:
                        cost_matrix[i, j] = 1.0  # High cost for no match
            
            # Apply Hungarian algorithm
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                for i, j in zip(row_indices, col_indices):
                    if score_matrix[i, j] >= threshold:
                        s_col = s_cols[i]
                        t_col = t_cols[j]
                        
                        # Get alternatives
                        alternatives = [s for s in table_scores 
                                      if s.source_column == s_col and s.target_column != t_col]
                        alternatives.sort(key=lambda x: x.combined_score, reverse=True)
                        
                        result = MappingResult(
                            source_table=s_table,
                            source_column=s_col,
                            target_table=t_table,
                            target_columns=[t_col],
                            mapping_type=MappingType.ONE_ONE,
                            scores=[score_matrix[i, j]],
                            confidence=score_matrix[i, j],
                            alternatives=alternatives[:5]  # Top 5 alternatives
                        )
                        results.append(result)
            except:
                # Fallback to greedy matching
                self._greedy_one_one_matching(table_scores, results, s_table, t_table, threshold)
        
        return results
    
    def _greedy_one_one_matching(
        self,
        table_scores: List[MappingScore],
        results: List[MappingResult],
        s_table: str,
        t_table: str,
        threshold: float
    ):
        """Fallback greedy matching for one-to-one mappings."""
        used_targets = set()
        sorted_scores = sorted(table_scores, key=lambda x: x.combined_score, reverse=True)
        
        for score in sorted_scores:
            if (score.target_column not in used_targets and 
                score.combined_score >= threshold):
                used_targets.add(score.target_column)
                
                alternatives = [s for s in table_scores 
                              if s.source_column == score.source_column and 
                              s.target_column != score.target_column]
                alternatives.sort(key=lambda x: x.combined_score, reverse=True)
                
                result = MappingResult(
                    source_table=s_table,
                    source_column=score.source_column,
                    target_table=t_table,
                    target_columns=[score.target_column],
                    mapping_type=MappingType.ONE_ONE,
                    scores=[score.combined_score],
                    confidence=score.combined_score,
                    alternatives=alternatives[:5]
                )
                results.append(result)
    
    def find_one_many_mappings(
        self,
        threshold: float = 0.5,
        max_targets: int = 3
    ) -> List[MappingResult]:
        """Find one-to-many mappings where one source column maps to multiple target columns."""
        results = []
        
        # Group by source column
        source_groups = {}
        for score in self.mapping_scores:
            key = (score.source_table, score.source_column)
            if key not in source_groups:
                source_groups[key] = []
            source_groups[key].append(score)
        
        for (s_table, s_col), scores in source_groups.items():
            # Filter by threshold and sort by score
            valid_scores = [s for s in scores if s.combined_score >= threshold]
            valid_scores.sort(key=lambda x: x.combined_score, reverse=True)
            
            if len(valid_scores) > 1:
                # Check if this could be a one-to-many mapping
                target_cols = [s.target_column for s in valid_scores[:max_targets]]
                target_scores = [s.combined_score for s in valid_scores[:max_targets]]
                
                # Only consider as one-to-many if scores are close
                if len(target_scores) > 1 and (max(target_scores) - min(target_scores)) < 0.2:
                    result = MappingResult(
                        source_table=s_table,
                        source_column=s_col,
                        target_table=valid_scores[0].target_table,
                        target_columns=target_cols,
                        mapping_type=MappingType.ONE_MANY,
                        scores=target_scores,
                        confidence=np.mean(target_scores),
                        alternatives=valid_scores[max_targets:max_targets+5]
                    )
                    results.append(result)
        
        return results
    
    def find_many_one_mappings(
        self,
        threshold: float = 0.5,
        max_sources: int = 3
    ) -> List[MappingResult]:
        """Find many-to-one mappings where multiple source columns map to one target column."""
        results = []
        
        # Group by target column
        target_groups = {}
        for score in self.mapping_scores:
            key = (score.target_table, score.target_column)
            if key not in target_groups:
                target_groups[key] = []
            target_groups[key].append(score)
        
        for (t_table, t_col), scores in target_groups.items():
            # Filter by threshold and sort by score
            valid_scores = [s for s in scores if s.combined_score >= threshold]
            valid_scores.sort(key=lambda x: x.combined_score, reverse=True)
            
            if len(valid_scores) > 1:
                # Check if this could be a many-to-one mapping
                source_cols = [s.source_column for s in valid_scores[:max_sources]]
                source_scores = [s.combined_score for s in valid_scores[:max_sources]]
                
                # Only consider as many-to-one if scores are close
                if len(source_scores) > 1 and (max(source_scores) - min(source_scores)) < 0.2:
                    result = MappingResult(
                        source_table=valid_scores[0].source_table,
                        source_column=", ".join(source_cols),  # Combined source
                        target_table=t_table,
                        target_columns=[t_col],
                        mapping_type=MappingType.MANY_ONE,
                        scores=source_scores,
                        confidence=np.mean(source_scores),
                        alternatives=valid_scores[max_sources:max_sources+5]
                    )
                    results.append(result)
        
        return results
    
    def find_many_many_mappings(
        self,
        threshold: float = 0.5,
        max_sources: int = 3,
        max_targets: int = 3
    ) -> List[MappingResult]:
        """Find many-to-many mappings using clustering approach."""
        results = []
        
        # Use graph-based approach to find clusters
        G = nx.Graph()
        
        # Add edges for high-scoring mappings
        for score in self.mapping_scores:
            if score.combined_score >= threshold:
                source_node = f"{score.source_table}.{score.source_column}"
                target_node = f"{score.target_table}.{score.target_column}"
                G.add_edge(source_node, target_node, weight=score.combined_score)
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        for component in components:
            if len(component) > 2:  # At least 2 nodes for many-to-many
                source_nodes = [n for n in component if "." in n and n.split(".")[0] in ["source"]]
                target_nodes = [n for n in component if "." in n and n.split(".")[0] in ["target"]]
                
                if len(source_nodes) > 1 and len(target_nodes) > 1:
                    # Extract column names
                    source_cols = [n.split(".", 1)[1] for n in source_nodes]
                    target_cols = [n.split(".", 1)[1] for n in target_nodes]
                    
                    # Calculate average score for this cluster
                    cluster_scores = []
                    for s_col in source_cols:
                        for t_col in target_cols:
                            edge_data = G.get_edge_data(f"source.{s_col}", f"target.{t_col}")
                            if edge_data:
                                cluster_scores.append(edge_data["weight"])
                    
                    if cluster_scores:
                        result = MappingResult(
                            source_table="multiple",
                            source_column=", ".join(source_cols[:max_sources]),
                            target_table="multiple",
                            target_columns=target_cols[:max_targets],
                            mapping_type=MappingType.MANY_MANY,
                            scores=cluster_scores[:max_sources * max_targets],
                            confidence=np.mean(cluster_scores),
                            alternatives=[]
                        )
                        results.append(result)
        
        return results
    
    def generate_mapping_report(
        self,
        source_tables: Dict[str, pd.DataFrame],
        target_tables: Dict[str, pd.DataFrame],
        table_pairs: List[List[str]],
        threshold: float = 0.5
    ) -> Dict[str, List[MappingResult]]:
        """Generate comprehensive mapping report with all mapping types."""
        
        # Calculate all scores first
        self.calculate_all_scores(source_tables, target_tables, {}, table_pairs)
        
        # Generate different mapping types
        one_one_mappings = self.optimize_one_one_mappings(
            source_tables, target_tables, table_pairs, threshold
        )
        
        one_many_mappings = self.find_one_many_mappings(threshold)
        many_one_mappings = self.find_many_one_mappings(threshold)
        many_many_mappings = self.find_many_many_mappings(threshold)
        
        return {
            "one_one": one_one_mappings,
            "one_many": one_many_mappings,
            "many_one": many_one_mappings,
            "many_many": many_many_mappings,
            "all_scores": self.mapping_scores
        }