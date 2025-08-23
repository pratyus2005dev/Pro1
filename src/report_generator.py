from __future__ import annotations
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from .mapping_engine import ColumnMappingEngine, MappingType
from .model import ColumnMappingModel
from .ddl_parser import load_ddl
from .data_loader import load_tables

class ReportGenerator:
    """Generate comprehensive mapping reports with visualizations."""
    
    def __init__(self, ddl_path: str, source_root: str, target_root: str, model_path: str):
        self.ddl_path = ddl_path
        self.source_root = source_root
        self.target_root = target_root
        self.model_path = model_path
        self.model = None
        self.engine = None
        
    def _load_model_and_engine(self):
        """Load the trained model and mapping engine."""
        if self.model is None:
            self.model = ColumnMappingModel.load(self.model_path)
            self.engine = ColumnMappingEngine(self.model)
    
    def generate_comprehensive_csv(
        self,
        table_pairs: List[List[str]],
        source_files: Dict[str, str],
        target_files: Dict[str, str],
        threshold: float = 0.5,
        output_path: str = "outputs/comprehensive_mappings.csv"
    ) -> str:
        """Generate comprehensive CSV with all mapping types and scores."""
        self._load_model_and_engine()
        
        # Load data
        ddl = load_ddl(self.ddl_path)
        source_tables = load_tables(self.source_root, source_files)
        target_tables = load_tables(self.target_root, target_files)
        
        # Generate mapping report
        mapping_report = self.engine.generate_mapping_report(
            source_tables, target_tables, table_pairs, threshold
        )
        
        # Create comprehensive DataFrame
        rows = []
        
        # Process one-to-one mappings
        for mapping in mapping_report["one_one"]:
            rows.append({
                "mapping_type": "one_one",
                "source_table": mapping.source_table,
                "source_column": mapping.source_column,
                "target_table": mapping.target_table,
                "target_columns": ", ".join(mapping.target_columns),
                "ml_score": mapping.scores[0] if mapping.scores else 0.0,
                "fuzzy_score": self._get_fuzzy_score(mapping),
                "combined_score": mapping.confidence,
                "confidence": mapping.confidence,
                "num_targets": len(mapping.target_columns),
                "num_sources": 1,
                "best_alternative_score": self._get_best_alternative_score(mapping),
                "score_rank": 1,
                "is_primary_mapping": True
            })
        
        # Process one-to-many mappings
        for mapping in mapping_report["one_many"]:
            for i, (target_col, score) in enumerate(zip(mapping.target_columns, mapping.scores)):
                rows.append({
                    "mapping_type": "one_many",
                    "source_table": mapping.source_table,
                    "source_column": mapping.source_column,
                    "target_table": mapping.target_table,
                    "target_columns": target_col,
                    "ml_score": score,
                    "fuzzy_score": self._get_fuzzy_score(mapping, target_col),
                    "combined_score": score,
                    "confidence": mapping.confidence,
                    "num_targets": len(mapping.target_columns),
                    "num_sources": 1,
                    "best_alternative_score": self._get_best_alternative_score(mapping),
                    "score_rank": i + 1,
                    "is_primary_mapping": i == 0
                })
        
        # Process many-to-one mappings
        for mapping in mapping_report["many_one"]:
            source_cols = mapping.source_column.split(", ")
            for i, (source_col, score) in enumerate(zip(source_cols, mapping.scores)):
                rows.append({
                    "mapping_type": "many_one",
                    "source_table": mapping.source_table,
                    "source_column": source_col,
                    "target_table": mapping.target_table,
                    "target_columns": ", ".join(mapping.target_columns),
                    "ml_score": score,
                    "fuzzy_score": self._get_fuzzy_score(mapping, source_col=source_col),
                    "combined_score": score,
                    "confidence": mapping.confidence,
                    "num_targets": len(mapping.target_columns),
                    "num_sources": len(source_cols),
                    "best_alternative_score": self._get_best_alternative_score(mapping),
                    "score_rank": i + 1,
                    "is_primary_mapping": i == 0
                })
        
        # Process many-to-many mappings
        for mapping in mapping_report["many_many"]:
            source_cols = mapping.source_column.split(", ")
            target_cols = mapping.target_columns
            score_idx = 0
            
            for i, source_col in enumerate(source_cols):
                for j, target_col in enumerate(target_cols):
                    if score_idx < len(mapping.scores):
                        score = mapping.scores[score_idx]
                        rows.append({
                            "mapping_type": "many_many",
                            "source_table": mapping.source_table,
                            "source_column": source_col,
                            "target_table": mapping.target_table,
                            "target_columns": target_col,
                            "ml_score": score,
                            "fuzzy_score": self._get_fuzzy_score(mapping, target_col, source_col),
                            "combined_score": score,
                            "confidence": mapping.confidence,
                            "num_targets": len(target_cols),
                            "num_sources": len(source_cols),
                            "best_alternative_score": self._get_best_alternative_score(mapping),
                            "score_rank": score_idx + 1,
                            "is_primary_mapping": i == 0 and j == 0
                        })
                        score_idx += 1
        
        # Create DataFrame and sort by combined score
        df = pd.DataFrame(rows)
        df = df.sort_values(['source_table', 'source_column', 'combined_score'], ascending=[True, True, False])
        
        # Add additional metadata
        df['timestamp'] = datetime.now().isoformat()
        df['model_type'] = self.model.model_type
        df['threshold_used'] = threshold
        
        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def _get_fuzzy_score(self, mapping, target_col: str = None, source_col: str = None) -> float:
        """Extract fuzzy score from mapping alternatives."""
        if not mapping.alternatives:
            return 0.0
        
        # Find the relevant alternative
        for alt in mapping.alternatives:
            if target_col and alt.target_column == target_col:
                return alt.fuzzy_score
            if source_col and alt.source_column == source_col:
                return alt.fuzzy_score
        
        # Return average fuzzy score if no specific match
        return np.mean([alt.fuzzy_score for alt in mapping.alternatives])
    
    def _get_best_alternative_score(self, mapping) -> float:
        """Get the best alternative score."""
        if not mapping.alternatives:
            return 0.0
        return max(alt.combined_score for alt in mapping.alternatives)
    
    def generate_summary_report(
        self,
        table_pairs: List[List[str]],
        source_files: Dict[str, str],
        target_files: Dict[str, str],
        threshold: float = 0.5,
        output_path: str = "outputs/mapping_summary.csv"
    ) -> str:
        """Generate a summary report with best mappings per source column."""
        self._load_model_and_engine()
        
        # Load data
        ddl = load_ddl(self.ddl_path)
        source_tables = load_tables(self.source_root, source_files)
        target_tables = load_tables(self.target_root, target_files)
        
        # Generate mapping report
        mapping_report = self.engine.generate_mapping_report(
            source_tables, target_tables, table_pairs, threshold
        )
        
        # Create summary DataFrame
        summary_rows = []
        
        for s_table, t_table in table_pairs:
            if s_table not in source_tables or t_table not in target_tables:
                continue
            
            for s_col in source_tables[s_table].columns:
                # Find best mapping for this source column
                best_mapping = None
                best_score = 0.0
                
                # Check one-to-one mappings
                for mapping in mapping_report["one_one"]:
                    if (mapping.source_table == s_table and 
                        mapping.source_column == s_col and
                        mapping.confidence > best_score):
                        best_mapping = mapping
                        best_score = mapping.confidence
                
                # Check one-to-many mappings
                for mapping in mapping_report["one_many"]:
                    if (mapping.source_table == s_table and 
                        mapping.source_column == s_col and
                        mapping.confidence > best_score):
                        best_mapping = mapping
                        best_score = mapping.confidence
                
                # Check many-to-one mappings
                for mapping in mapping_report["many_one"]:
                    source_cols = mapping.source_column.split(", ")
                    if (mapping.source_table == s_table and 
                        s_col in source_cols and
                        mapping.confidence > best_score):
                        best_mapping = mapping
                        best_score = mapping.confidence
                
                # Check many-to-many mappings
                for mapping in mapping_report["many_many"]:
                    source_cols = mapping.source_column.split(", ")
                    if (mapping.source_table == s_table and 
                        s_col in source_cols and
                        mapping.confidence > best_score):
                        best_mapping = mapping
                        best_score = mapping.confidence
                
                if best_mapping:
                    summary_rows.append({
                        "source_table": s_table,
                        "source_column": s_col,
                        "target_table": best_mapping.target_table,
                        "best_target_columns": ", ".join(best_mapping.target_columns),
                        "mapping_type": best_mapping.mapping_type.value,
                        "best_score": best_score,
                        "ml_score": best_mapping.scores[0] if best_mapping.scores else 0.0,
                        "fuzzy_score": self._get_fuzzy_score(best_mapping),
                        "confidence": best_mapping.confidence,
                        "num_alternatives": len(best_mapping.alternatives),
                        "is_mapped": best_score >= threshold
                    })
                else:
                    summary_rows.append({
                        "source_table": s_table,
                        "source_column": s_col,
                        "target_table": "",
                        "best_target_columns": "",
                        "mapping_type": "unmapped",
                        "best_score": 0.0,
                        "ml_score": 0.0,
                        "fuzzy_score": 0.0,
                        "confidence": 0.0,
                        "num_alternatives": 0,
                        "is_mapped": False
                    })
        
        # Create DataFrame
        df = pd.DataFrame(summary_rows)
        df = df.sort_values(['source_table', 'best_score'], ascending=[True, False])
        
        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_visualizations(
        self,
        csv_path: str,
        output_dir: str = "outputs/visualizations"
    ):
        """Generate visualizations for the mapping results."""
        df = pd.read_csv(csv_path)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Score distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['combined_score'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Combined Scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.scatter(df['fuzzy_score'], df['ml_score'], alpha=0.6)
        plt.title('Fuzzy Score vs ML Score')
        plt.xlabel('Fuzzy Score')
        plt.ylabel('ML Score')
        
        plt.subplot(2, 2, 3)
        mapping_counts = df['mapping_type'].value_counts()
        plt.pie(mapping_counts.values, labels=mapping_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Mapping Types')
        
        plt.subplot(2, 2, 4)
        plt.hist(df['confidence'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mapping_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Score correlation heatmap
        numeric_cols = ['combined_score', 'ml_score', 'fuzzy_score', 'confidence']
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Score Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mapping type performance
        plt.figure(figsize=(10, 6))
        mapping_performance = df.groupby('mapping_type')['combined_score'].agg(['mean', 'std', 'count'])
        
        x = range(len(mapping_performance))
        plt.bar(x, mapping_performance['mean'], yerr=mapping_performance['std'], capsize=5)
        plt.xticks(x, mapping_performance.index, rotation=45)
        plt.title('Average Score by Mapping Type')
        plt.ylabel('Average Combined Score')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mapping_type_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(
        self,
        table_pairs: List[List[str]],
        source_files: Dict[str, str],
        target_files: Dict[str, str],
        threshold: float = 0.5,
        output_format: str = "csv",
        include_visualizations: bool = True
    ) -> str:
        """Generate comprehensive mapping report."""
        
        # Generate comprehensive CSV
        csv_path = self.generate_comprehensive_csv(
            table_pairs, source_files, target_files, threshold
        )
        
        # Generate summary CSV
        summary_path = self.generate_summary_report(
            table_pairs, source_files, target_files, threshold
        )
        
        # Generate visualizations if requested
        if include_visualizations:
            self.generate_visualizations(csv_path)
        
        # Generate JSON report if requested
        if output_format == "json":
            json_path = csv_path.replace('.csv', '.json')
            df = pd.read_csv(csv_path)
            df.to_json(json_path, orient='records', indent=2)
            return json_path
        
        # Generate Excel report if requested
        elif output_format == "excel":
            excel_path = csv_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Comprehensive mappings
                df_comprehensive = pd.read_csv(csv_path)
                df_comprehensive.to_excel(writer, sheet_name='Comprehensive_Mappings', index=False)
                
                # Summary mappings
                df_summary = pd.read_csv(summary_path)
                df_summary.to_excel(writer, sheet_name='Summary_Mappings', index=False)
                
                # Statistics
                stats_data = {
                    'Metric': [
                        'Total Mappings',
                        'One-to-One Mappings',
                        'One-to-Many Mappings',
                        'Many-to-One Mappings',
                        'Many-to-Many Mappings',
                        'Average Combined Score',
                        'Average ML Score',
                        'Average Fuzzy Score',
                        'High Confidence Mappings (>0.8)',
                        'Medium Confidence Mappings (0.5-0.8)',
                        'Low Confidence Mappings (<0.5)'
                    ],
                    'Value': [
                        len(df_comprehensive),
                        len(df_comprehensive[df_comprehensive['mapping_type'] == 'one_one']),
                        len(df_comprehensive[df_comprehensive['mapping_type'] == 'one_many']),
                        len(df_comprehensive[df_comprehensive['mapping_type'] == 'many_one']),
                        len(df_comprehensive[df_comprehensive['mapping_type'] == 'many_many']),
                        df_comprehensive['combined_score'].mean(),
                        df_comprehensive['ml_score'].mean(),
                        df_comprehensive['fuzzy_score'].mean(),
                        len(df_comprehensive[df_comprehensive['confidence'] > 0.8]),
                        len(df_comprehensive[(df_comprehensive['confidence'] > 0.5) & (df_comprehensive['confidence'] <= 0.8)]),
                        len(df_comprehensive[df_comprehensive['confidence'] <= 0.5])
                    ]
                }
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            return excel_path
        
        return csv_path