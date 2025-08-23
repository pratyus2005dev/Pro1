from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from joblib import dump, load

class ColumnMappingModel:
    """Enhanced column mapping model with multiple algorithms and ensemble methods."""
    
    def __init__(self, model_type: str = "ensemble", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.best_params = None
        
    def build_model(self, model_type: str = None) -> Any:
        """Build the specified model type."""
        if model_type:
            self.model_type = model_type
            
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        elif self.model_type == "catboost":
            self.model = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_type == "svm":
            self.model = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            )
        elif self.model_type == "ensemble":
            # Create ensemble of best performing models
            models = [
                ('xgb', xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    use_label_encoder=False
                )),
                ('lgb', lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    verbose=-1
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=self.random_state
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state
                ))
            ]
            self.model = VotingClassifier(
                estimators=models,
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return self.model
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning for the selected model."""
        if self.model_type == "xgboost":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == "lightgbm":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            # For other models, use default parameters
            return {"best_params": None, "best_score": 0.0}
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return {
            "best_params": self.best_params,
            "best_score": grid_search.best_score_
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """Fit the model and optionally perform hyperparameter tuning."""
        if self.model is None:
            self.build_model()
        
        if tune_hyperparameters:
            tuning_results = self.hyperparameter_tuning(X, y)
        else:
            self.model.fit(X, y)
            tuning_results = {"best_params": None, "best_score": 0.0}
        
        # Calculate feature importance if available
        self.feature_importance = self._get_feature_importance()
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        
        return {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "tuning_results": tuning_results,
            "feature_importance": self.feature_importance
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if self.model is None:
            return None
            
        try:
            if hasattr(self.model, 'feature_importances_'):
                return dict(enumerate(self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                return dict(enumerate(np.abs(self.model.coef_[0])))
            else:
                return None
        except:
            return None
    
    def save(self, filepath: str):
        """Save the model to disk."""
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance,
            "best_params": self.best_params
        }
        dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ColumnMappingModel':
        """Load a model from disk."""
        model_data = load(filepath)
        instance = cls(model_type=model_data["model_type"])
        instance.model = model_data["model"]
        instance.feature_importance = model_data["feature_importance"]
        instance.best_params = model_data["best_params"]
        return instance

def build_model(model_type: str = "ensemble", random_state: int = 42) -> ColumnMappingModel:
    """Factory function to build a column mapping model."""
    return ColumnMappingModel(model_type=model_type, random_state=random_state)

def evaluate_model(model: ColumnMappingModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    return {
        "auc": auc,
        "average_precision": ap,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
