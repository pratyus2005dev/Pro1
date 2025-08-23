from __future__ import annotations
import json
from typing import Dict, List
import pandas as pd
import joblib
from .ddl_parser import load_ddl
from .data_loader import load_tables
from .enhanced_featurizer import EnhancedFeaturizer
from .enhanced_model import build_model
from .mapping_engine import MappingEngine
import numpy as np

def predict_mappings(
    ddl_path: str,
    source_root: str, source_files: Dict[str, str],
    target_root: str, target_files: Dict[str, str],
    table_pairs: List[List[str]],
    model_in: str,
    featurizer_in: str,
    threshold: float = 0.5,
    out_csv: str = "outputs/enhanced_mapping_suggestions.csv",
    out_json: str = "outputs/enhanced_mapping_suggestions.json",
    out_detailed_csv: str = "outputs/detailed_mapping_analysis.csv"
):
    """Enhanced prediction with comprehensive mapping analysis"""
    
    print("Loading model and featurizer...")
    model = build_model().load(model_in)
    featurizer = joblib.load(featurizer_in)
    
    print("Loading DDL and data...")
    ddl = load_ddl(ddl_path)
    src = load_tables(source_root, source_files)
    tgt = load_tables(target_root, target_files)
    
    print("Initializing mapping engine...")
    mapping_engine = MappingEngine(featurizer, model, ddl)
    
    print("Generating comprehensive mappings...")
    mappings = mapping_engine.generate_final_mappings(
        src, tgt, table_pairs, threshold=threshold
    )
    
    # Export results
    print("Exporting results...")
    
    # Main CSV with best mappings
    main_df = mapping_engine.export_to_csv(mappings, out_csv)
    
    # Detailed JSON with all information
    detailed_json = mapping_engine.export_to_json(mappings, out_json)
    
    # Detailed CSV analysis
    detailed_df = create_detailed_analysis(mappings, out_detailed_csv)
    
    # Generate summary statistics
    summary = generate_mapping_summary(mappings)
    
    print("Prediction completed successfully!")
    print(f"Results saved to:")
    print(f"  - Main mappings: {out_csv}")
    print(f"  - Detailed JSON: {out_json}")
    print(f"  - Detailed analysis: {out_detailed_csv}")
    
    return {
        "summary": summary,
        "main_csv": out_csv,
        "detailed_json": out_json,
        "detailed_csv": out_detailed_csv,
        "mapping_counts": {
            "one_to_one": len(mappings['one_to_one']),
            "one_to_many": len(mappings['one_to_many']),
            "many_to_one": len(mappings['many_to_one']),
            "many_to_many": len(mappings['many_to_many']),
            "total_candidates": len(mappings['all_candidates'])
        }
    }

def create_detailed_analysis(mappings: Dict[str, List], output_path: str) -> pd.DataFrame:
    """Create detailed analysis DataFrame with all mapping information"""
    
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
                
                # Fuzzy matching scores
                'levenshtein_ratio': mapping.metadata.get('levenshtein_ratio', 0),
                'partial_ratio': mapping.metadata.get('partial_ratio', 0),
                'token_sort_ratio': mapping.metadata.get('token_sort_ratio', 0),
                'token_set_ratio': mapping.metadata.get('token_set_ratio', 0),
                'jaro_winkler': mapping.metadata.get('jaro_winkler', 0),
                'sequence_ratio': mapping.metadata.get('sequence_ratio', 0),
                'token_jaccard': mapping.metadata.get('token_jaccard', 0),
                
                # Semantic and type features
                'semantic_similarity': mapping.metadata.get('semantic_similarity', 0),
                'coarse_type_match': mapping.metadata.get('coarse_type_match', 0),
                
                # Data profiling features
                'null_rate_diff': mapping.metadata.get('null_rate_diff', 0),
                'cardinality_ratio': mapping.metadata.get('cardinality_ratio', 0),
                'value_overlap': mapping.metadata.get('value_overlap', 0),
                'length_similarity': mapping.metadata.get('length_similarity', 0),
                
                # Type-specific features
                'strlen_med_diff': mapping.metadata.get('strlen_med_diff', 0),
                'string_val_jaccard': mapping.metadata.get('string_val_jaccard', 0),
                'numeric_stat_sim': mapping.metadata.get('numeric_stat_sim', 0),
                'date_overlap': mapping.metadata.get('date_overlap', 0),
                
                # Metadata features
                'pk_similarity': mapping.metadata.get('pk_similarity', 0),
                'fk_similarity': mapping.metadata.get('fk_similarity', 0),
                'unique_similarity': mapping.metadata.get('unique_similarity', 0),
                'nullable_similarity': mapping.metadata.get('nullable_similarity', 0),
                'default_similarity': mapping.metadata.get('default_similarity', 0),
                'type_similarity': mapping.metadata.get('type_similarity', 0),
                
                # Quality indicators
                'high_confidence': mapping.confidence >= 0.8,
                'exact_name_match': mapping.metadata.get('levenshtein_ratio', 0) >= 0.9,
                'type_compatible': mapping.metadata.get('coarse_type_match', 0) == 1.0,
                'semantic_match': mapping.metadata.get('semantic_similarity', 0) >= 0.7
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add quality score
    df['quality_score'] = (
        df['combined_score'] * 0.4 +
        df['confidence'] * 0.3 +
        df['semantic_similarity'] * 0.2 +
        df['coarse_type_match'] * 0.1
    )
    
    # Sort by quality score
    df = df.sort_values('quality_score', ascending=False)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

def generate_mapping_summary(mappings: Dict[str, List]) -> Dict:
    """Generate comprehensive mapping summary statistics"""
    
    summary = {
        "total_mappings": 0,
        "mapping_types": {},
        "score_distributions": {},
        "quality_metrics": {},
        "table_coverage": {}
    }
    
    # Count mappings by type
    for mapping_type, mapping_list in mappings.items():
        if mapping_type == 'all_candidates':
            continue
            
        count = len(mapping_list)
        summary["total_mappings"] += count
        summary["mapping_types"][mapping_type] = count
        
        if mapping_list:
            # Score distributions
            scores = {
                'ml_scores': [m.ml_score for m in mapping_list],
                'fuzzy_scores': [m.fuzzy_score for m in mapping_list],
                'combined_scores': [m.combined_score for m in mapping_list],
                'confidence_scores': [m.confidence for m in mapping_list]
            }
            
            summary["score_distributions"][mapping_type] = {
                metric: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
                for metric, values in scores.items()
            }
            
            # Quality metrics
            high_confidence = sum(1 for m in mapping_list if m.confidence >= 0.8)
            high_score = sum(1 for m in mapping_list if m.combined_score >= 0.8)
            type_compatible = sum(1 for m in mapping_list if m.metadata.get('coarse_type_match', 0) == 1.0)
            
            summary["quality_metrics"][mapping_type] = {
                'high_confidence_ratio': high_confidence / count if count > 0 else 0,
                'high_score_ratio': high_score / count if count > 0 else 0,
                'type_compatible_ratio': type_compatible / count if count > 0 else 0
            }
    
    # Table coverage analysis
    source_tables = set()
    target_tables = set()
    
    for mapping_type, mapping_list in mappings.items():
        if mapping_type == 'all_candidates':
            continue
            
        for mapping in mapping_list:
            source_tables.add(mapping.source_table)
            target_tables.add(mapping.target_table)
    
    summary["table_coverage"] = {
        'source_tables': list(source_tables),
        'target_tables': list(target_tables),
        'source_table_count': len(source_tables),
        'target_table_count': len(target_tables)
    }
    
    return summary

def analyze_mapping_quality(mappings: Dict[str, List]) -> pd.DataFrame:
    """Analyze mapping quality and provide recommendations"""
    
    quality_analysis = []
    
    for mapping_type, mapping_list in mappings.items():
        if mapping_type == 'all_candidates':
            continue
            
        for mapping in mapping_list:
            # Quality indicators
            quality_indicators = {
                'excellent_match': (
                    mapping.combined_score >= 0.9 and 
                    mapping.confidence >= 0.9 and
                    mapping.metadata.get('coarse_type_match', 0) == 1.0
                ),
                'good_match': (
                    mapping.combined_score >= 0.7 and 
                    mapping.confidence >= 0.7
                ),
                'needs_review': (
                    mapping.combined_score < 0.5 or 
                    mapping.confidence < 0.5
                ),
                'type_mismatch': (
                    mapping.metadata.get('coarse_type_match', 0) == 0.0 and
                    mapping.combined_score > 0.7
                )
            }
            
            # Recommendations
            recommendations = []
            if quality_indicators['excellent_match']:
                recommendations.append("High confidence mapping - recommended for production")
            elif quality_indicators['good_match']:
                recommendations.append("Good match - review recommended")
            elif quality_indicators['needs_review']:
                recommendations.append("Low confidence - manual review required")
            elif quality_indicators['type_mismatch']:
                recommendations.append("Type mismatch detected - verify data compatibility")
            
            if mapping.metadata.get('semantic_similarity', 0) < 0.3:
                recommendations.append("Low semantic similarity - verify business logic")
            
            analysis_row = {
                'mapping_type': mapping_type,
                'source_table': mapping.source_table,
                'source_column': mapping.source_column,
                'target_table': mapping.target_table,
                'target_column': mapping.target_column,
                'combined_score': mapping.combined_score,
                'confidence': mapping.confidence,
                'quality_category': (
                    'excellent' if quality_indicators['excellent_match']
                    else 'good' if quality_indicators['good_match']
                    else 'needs_review' if quality_indicators['needs_review']
                    else 'type_mismatch' if quality_indicators['type_mismatch']
                    else 'moderate'
                ),
                'recommendations': '; '.join(recommendations) if recommendations else 'No specific recommendations',
                'risk_level': (
                    'low' if quality_indicators['excellent_match']
                    else 'medium' if quality_indicators['good_match']
                    else 'high'
                )
            }
            
            quality_analysis.append(analysis_row)
    
    return pd.DataFrame(quality_analysis)

def export_mapping_report(
    mappings: Dict[str, List],
    output_dir: str = "outputs"
):
    """Generate comprehensive mapping report"""
    
    # Create detailed analysis
    detailed_df = create_detailed_analysis(mappings, f"{output_dir}/detailed_mapping_analysis.csv")
    
    # Create quality analysis
    quality_df = analyze_mapping_quality(mappings)
    quality_df.to_csv(f"{output_dir}/mapping_quality_analysis.csv", index=False)
    
    # Generate summary
    summary = generate_mapping_summary(mappings)
    
    # Save summary to JSON
    with open(f"{output_dir}/mapping_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create HTML report
    create_html_report(detailed_df, quality_df, summary, f"{output_dir}/mapping_report.html")
    
    return {
        'detailed_analysis': f"{output_dir}/detailed_mapping_analysis.csv",
        'quality_analysis': f"{output_dir}/mapping_quality_analysis.csv",
        'summary': f"{output_dir}/mapping_summary.json",
        'html_report': f"{output_dir}/mapping_report.html"
    }

def create_html_report(detailed_df: pd.DataFrame, quality_df: pd.DataFrame, summary: Dict, output_path: str):
    """Create HTML report with mapping analysis"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Schema Mapping Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .excellent {{ background-color: #d4edda; }}
            .good {{ background-color: #d1ecf1; }}
            .needs_review {{ background-color: #fff3cd; }}
            .type_mismatch {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Schema Mapping Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <strong>Total Mappings:</strong> {summary['total_mappings']}
            </div>
            <div class="metric">
                <strong>Source Tables:</strong> {summary['table_coverage']['source_table_count']}
            </div>
            <div class="metric">
                <strong>Target Tables:</strong> {summary['table_coverage']['target_table_count']}
            </div>
        </div>
        
        <div class="section">
            <h2>Mapping Distribution by Type</h2>
            <table>
                <tr><th>Mapping Type</th><th>Count</th></tr>
                {''.join(f'<tr><td>{mtype}</td><td>{count}</td></tr>' for mtype, count in summary['mapping_types'].items())}
            </table>
        </div>
        
        <div class="section">
            <h2>Quality Analysis</h2>
            <table>
                <tr>
                    <th>Source Table</th>
                    <th>Source Column</th>
                    <th>Target Table</th>
                    <th>Target Column</th>
                    <th>Quality Category</th>
                    <th>Risk Level</th>
                    <th>Combined Score</th>
                </tr>
                {''.join(f'''
                <tr class="{row['quality_category']}">
                    <td>{row['source_table']}</td>
                    <td>{row['source_column']}</td>
                    <td>{row['target_table']}</td>
                    <td>{row['target_column']}</td>
                    <td>{row['quality_category']}</td>
                    <td>{row['risk_level']}</td>
                    <td>{row['combined_score']:.3f}</td>
                </tr>
                ''' for _, row in quality_df.head(20).iterrows())}
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)