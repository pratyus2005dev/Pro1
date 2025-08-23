from __future__ import annotations
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

class ModelSelector:
    """Model selection and ensemble framework"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.model_scores = {}
        
    def build_models(self) -> Dict[str, Any]:
        """Build all available models"""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=self.random_state
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            random_state=self.random_state
        )
        
        # XGBoost
        if _HAS_XGB:
            models['xgboost'] = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.5,
                reg_alpha=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state
            )
        
        # LightGBM
        if _HAS_LGBM:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.5,
                reg_alpha=0.0,
                objective="binary",
                metric="binary_logloss",
                random_state=self.random_state
            )
        
        # CatBoost
        if _HAS_CATBOOST:
            models['catboost'] = CatBoostClassifier(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bylevel=0.8,
                reg_lambda=1.5,
                random_state=self.random_state,
                verbose=False
            )
        
        # SVM
        models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        # Logistic Regression
        models['logistic'] = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # Neural Network
        models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state
        )
        
        self.models = models
        return models
    
    def select_best_model(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Tuple[str, Any]:
        """Select the best performing model using cross-validation"""
        if not self.models:
            self.build_models()
        
        best_model_name = None
        best_score = 0.0
        
        for name, model in self.models.items():
            try:
                # Use ROC-AUC for model selection
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
                mean_score = scores.mean()
                self.model_scores[name] = {
                    'mean_score': mean_score,
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    
            except Exception as e:
                print(f"Error with model {name}: {e}")
                self.model_scores[name] = {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}
        
        self.best_model = self.models[best_model_name]
        self.best_score = best_score
        
        return best_model_name, self.best_model
    
    def build_ensemble(self, X: np.ndarray, y: np.ndarray, top_k: int = 3) -> VotingClassifier:
        """Build ensemble from top-k performing models"""
        if not self.model_scores:
            self.select_best_model(X, y)
        
        # Sort models by performance
        sorted_models = sorted(
            self.model_scores.items(),
            key=lambda x: x[1]['mean_score'],
            reverse=True
        )
        
        # Select top-k models
        top_models = sorted_models[:top_k]
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[score['mean_score'] for _, score in top_models]
        )
        
        return ensemble
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Any:
        """Perform hyperparameter tuning for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [200, 300, 400],
                'max_depth': [8, 12, 16],
                'min_samples_split': [2, 4, 8]
            },
            'gradient_boosting': {
                'n_estimators': [200, 300, 400],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'xgboost': {
                'n_estimators': [200, 300, 400],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'reg_lambda': [0.5, 1.0, 1.5]
            },
            'lightgbm': {
                'n_estimators': [200, 300, 400],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'reg_lambda': [0.5, 1.0, 1.5]
            }
        }
        
        if model_name not in param_grids:
            return self.models[model_name]
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_

class EnhancedModel:
    """Enhanced model wrapper with multiple algorithms and ensemble support"""
    
    def __init__(self, random_state: int = 42, use_ensemble: bool = True):
        self.random_state = random_state
        self.use_ensemble = use_ensemble
        self.selector = ModelSelector(random_state)
        self.final_model = None
        self.model_info = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """Train the model with automatic selection"""
        
        # Select best individual model
        best_name, best_model = self.selector.select_best_model(X, y)
        
        if tune_hyperparameters:
            best_model = self.selector.hyperparameter_tuning(X, y, best_name)
        
        # Train the best individual model
        best_model.fit(X, y)
        
        # Build ensemble if requested
        if self.use_ensemble:
            ensemble = self.selector.build_ensemble(X, y)
            ensemble.fit(X, y)
            self.final_model = ensemble
            model_type = "ensemble"
        else:
            self.final_model = best_model
            model_type = best_name
        
        # Evaluate performance
        y_pred_proba = self.final_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        ap = average_precision_score(y, y_pred_proba)
        
        self.model_info = {
            'model_type': model_type,
            'best_individual_model': best_name,
            'auc_score': auc,
            'ap_score': ap,
            'model_scores': self.selector.model_scores,
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        }
        
        return self.model_info
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.final_model is None:
            raise ValueError("Model not trained yet")
        return self.final_model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def save(self, filepath: str):
        """Save the model and metadata"""
        model_data = {
            'model': self.final_model,
            'model_info': self.model_info,
            'selector': self.selector
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'EnhancedModel':
        """Load a saved model"""
        model_data = joblib.load(filepath)
        instance = cls()
        instance.final_model = model_data['model']
        instance.model_info = model_data['model_info']
        instance.selector = model_data['selector']
        return instance

def build_model(random_state: int = 42, use_ensemble: bool = True) -> EnhancedModel:
    """Factory function to build enhanced model"""
    return EnhancedModel(random_state=random_state, use_ensemble=use_ensemble)