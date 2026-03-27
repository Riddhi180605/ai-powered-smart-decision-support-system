"""
AI-Powered Reasoning and Advisory Engine

This module provides intelligent business insights using external LLM APIs
and SHAP (SHapley Additive exPlanations) for model interpretability.

Features:
- Interpret ML model predictions
- Analyze SHAP feature importance
- Compare with historical trends
- Generate actionable business recommendations
- Provide human-like business insights
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# ML and SHAP imports
import shap
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAdvisor:
    """
    AI-Powered Reasoning and Advisory Engine

    Provides intelligent business insights by combining:
    1. ML model predictions
    2. SHAP explanations
    3. Historical data analysis
    4. LLM-powered natural language insights
    """

    def __init__(self, llm_api_key: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize the AI Advisor

        Args:
            llm_api_key: API key for configured LLM provider
            provider: LLM provider name (groq, openai). If omitted, inferred from env keys.
        """
        self.provider = self._resolve_provider(provider)
        self.api_key = self._resolve_llm_api_key(llm_api_key, self.provider)
        if not self.api_key:
            env_var = self._provider_api_env(self.provider)
            raise ValueError(
                f"{self.provider.upper()} API key is required. Set {env_var} in the same shell that starts the backend, "
                f"or add {env_var} to backend/.env or project .env."
            )

        base_url = self._resolve_base_url(self.provider)
        self.client = OpenAI(api_key=self.api_key, base_url=base_url) if base_url else OpenAI(api_key=self.api_key)
        self.llm_models = self._resolve_llm_models(self.provider)
        self.shap_explainer = None
        self.scaler = StandardScaler()

    @staticmethod
    def _sanitize_api_key(raw_key: Optional[str]) -> str:
        """Normalize API key values from env files and request payloads."""
        if raw_key is None:
            return ""

        key = str(raw_key).strip()
        if not key:
            return ""

        # Handle accidental wrapping quotes in .env or manual input.
        if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
            key = key[1:-1].strip()

        return key

    @staticmethod
    def _provider_api_env(provider: str) -> str:
        provider = (provider or '').strip().lower()
        if provider == 'openai':
            return 'OPENAI_API_KEY'
        return 'GROQ_API_KEY'

    @classmethod
    def _resolve_provider(cls, provider: Optional[str]) -> str:
        """Resolve provider from argument or env. Defaults to groq."""
        explicit_provider = (provider or '').strip().lower()
        if explicit_provider:
            return explicit_provider

        env_provider = os.getenv('LLM_PROVIDER', '').strip().lower()
        if env_provider:
            return env_provider

        return 'groq'

    @classmethod
    def _resolve_llm_api_key(cls, llm_api_key: Optional[str], provider: str) -> str:
        """Resolve provider API key from parameter, environment variable, or local .env files."""
        param_key = cls._sanitize_api_key(llm_api_key)
        if param_key:
            return param_key

        primary_env = cls._provider_api_env(provider)
        env_key = cls._sanitize_api_key(os.getenv(primary_env))
        if env_key:
            return env_key

        file_key = cls._load_api_key_from_env_files([primary_env])
        if file_key:
            os.environ[primary_env] = file_key
            return file_key

        return ""

    @classmethod
    def _load_api_key_from_env_files(cls, key_names: List[str]) -> str:
        """Try reading one of key_names from backend/.env, then project .env."""
        env_paths = [
            Path(__file__).resolve().parent / '.env',
            Path(__file__).resolve().parent.parent / '.env',
        ]

        key_set = {k.strip() for k in key_names if k and k.strip()}

        for env_path in env_paths:
            if not env_path.exists():
                continue

            try:
                for line in env_path.read_text(encoding='utf-8').splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue

                    stripped = stripped.lstrip('\ufeff')

                    if stripped.startswith('export '):
                        stripped = stripped[len('export '):].strip()

                    if '=' not in stripped:
                        continue

                    name, value = stripped.split('=', 1)
                    if name.strip() in key_set:
                        value = value.split(' #', 1)[0].strip()
                        parsed_key = cls._sanitize_api_key(value)
                        if parsed_key:
                            return parsed_key
            except Exception as e:
                logger.warning(f"Failed reading env file {env_path}: {e}")

        return ""

    @staticmethod
    def _resolve_base_url(provider: str) -> Optional[str]:
        """Resolve OpenAI-compatible base URL for configured provider."""
        explicit = os.getenv('LLM_BASE_URL', '').strip()
        if explicit:
            return explicit

        if provider == 'groq':
            return 'https://api.groq.com/openai/v1'

        return None

    @staticmethod
    def _resolve_llm_models(provider: str) -> List[str]:
        """Resolve preferred models from env or provider defaults."""
        configured = os.getenv('LLM_MODELS', '').strip()
        if configured:
            models = [m.strip() for m in configured.split(',') if m.strip()]
            if models:
                return models

        single_model = os.getenv('LLM_MODEL', '').strip()
        if single_model:
            return [single_model]

        if provider == 'openai':
            return ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"]

        # Groq defaults.
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

    def analyze_predictions(
        self,
        model: BaseEstimator,
        X_data: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        predictions: Optional[np.ndarray] = None,
        target_column: str = "",
        dataset_type: str = "General"
    ) -> Dict[str, Any]:
        """
        Analyze model predictions with SHAP explanations

        Args:
            model: Trained ML model
            X_data: Feature data
            y_true: True target values (optional)
            predictions: Model predictions (optional, will be generated if not provided)
            target_column: Name of target column
            dataset_type: Type of dataset (Sales, HR, Customer, etc.)

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Generate predictions if not provided
            if predictions is None:
                # Scale data if needed
                X_scaled = self.scaler.fit_transform(X_data)
                predictions = model.predict(X_scaled)

            # Create SHAP explainer
            self._create_shap_explainer(model, X_data)

            # Calculate SHAP values
            shap_values = self._calculate_shap_values(X_data)

            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(X_data, shap_values)

            # Generate insights
            insights = self._generate_business_insights(
                model=model,
                X_data=X_data,
                predictions=predictions,
                y_true=y_true,
                shap_values=shap_values,
                feature_importance=feature_importance,
                target_column=target_column,
                dataset_type=dataset_type
            )

            return {
                "success": True,
                "feature_importance": feature_importance,
                "insights": insights,
                "shap_summary": self._summarize_shap_values(shap_values, X_data),
                "predictions_summary": self._summarize_predictions(predictions, y_true)
            }

        except Exception as e:
            logger.error(f"Error in analyze_predictions: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_shap_explainer(self, model: BaseEstimator, X_data: pd.DataFrame) -> None:
        """Create SHAP explainer for the model"""
        try:
            # Scale data for SHAP
            X_scaled = self.scaler.fit_transform(X_data)

            # Choose appropriate explainer based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'coef_'):
                # Linear models
                self.shap_explainer = shap.LinearExplainer(model, X_scaled)
            else:
                # Kernel explainer as fallback
                background = X_scaled[np.random.choice(X_scaled.shape[0], size=min(100, X_scaled.shape[0]), replace=False)]
                self.shap_explainer = shap.KernelExplainer(model.predict, background)

        except Exception as e:
            logger.warning(f"Could not create TreeExplainer, falling back to KernelExplainer: {str(e)}")
            # Fallback to KernelExplainer
            background = self.scaler.transform(X_data)
            background = background[np.random.choice(background.shape[0], size=min(50, background.shape[0]), replace=False)]
            self.shap_explainer = shap.KernelExplainer(model.predict, background)

    def _calculate_shap_values(self, X_data: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for the dataset"""
        X_scaled = self.scaler.transform(X_data)

        # Calculate SHAP values for a sample (limit for performance)
        sample_size = min(100, len(X_data))
        sample_indices = np.random.choice(len(X_data), size=sample_size, replace=False)
        X_sample = X_scaled[sample_indices]

        shap_values = self.shap_explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class classification
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values

        return shap_values

    def _analyze_feature_importance(self, X_data: pd.DataFrame, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze and rank feature importance based on SHAP values"""
        feature_names = X_data.columns.tolist()

        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance list
        feature_importance = []
        for i, feature in enumerate(feature_names):
            importance_score = float(mean_shap[i]) if i < len(mean_shap) else 0.0

            feature_importance.append({
                "feature": feature,
                "importance": importance_score,
                "rank": 0  # Will be set after sorting
            })

        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # Set ranks
        for i, feat in enumerate(feature_importance):
            feat["rank"] = i + 1

        return feature_importance

    def _generate_business_insights(
        self,
        model: BaseEstimator,
        X_data: pd.DataFrame,
        predictions: np.ndarray,
        y_true: Optional[pd.Series],
        shap_values: np.ndarray,
        feature_importance: List[Dict[str, Any]],
        target_column: str,
        dataset_type: str
    ) -> List[Dict[str, Any]]:
        """Generate business insights using configured LLM provider"""

        # Prepare context for LLM
        context = self._prepare_insight_context(
            X_data=X_data,
            predictions=predictions,
            y_true=y_true,
            feature_importance=feature_importance,
            target_column=target_column,
            dataset_type=dataset_type
        )

        # Generate insights using LLM
        insights = []

        # Generate different types of insights
        insight_types = [
            "prediction_interpretation",
            "feature_analysis",
            "trend_comparison",
            "actionable_recommendations"
        ]

        for insight_type in insight_types:
            try:
                insight = self._generate_single_insight(context, insight_type)
                if insight:
                    insights.append({
                        "type": insight_type,
                        "content": insight,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.warning(f"Failed to generate {insight_type} insight: {str(e)}")

        return insights

    def _prepare_insight_context(
        self,
        X_data: pd.DataFrame,
        predictions: np.ndarray,
        y_true: Optional[pd.Series],
        feature_importance: List[Dict[str, Any]],
        target_column: str,
        dataset_type: str
    ) -> Dict[str, Any]:
        """Prepare context data for LLM insight generation"""

        # Basic statistics
        context = {
            "dataset_type": dataset_type,
            "target_column": target_column,
            "num_samples": len(X_data),
            "num_features": len(X_data.columns),
            "feature_columns": X_data.columns.tolist()
        }

        # Prediction statistics
        if predictions is not None:
            context["predictions"] = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "sample_predictions": predictions[:10].tolist()  # Sample of predictions
            }

        # True values statistics (if available)
        if y_true is not None:
            context["true_values"] = {
                "mean": float(y_true.mean()),
                "std": float(y_true.std()),
                "min": float(y_true.min()),
                "max": float(y_true.max())
            }

        # Top 5 important features
        context["top_features"] = feature_importance[:5]

        # Feature correlations (if target available)
        if y_true is not None:
            correlations = {}
            for col in X_data.select_dtypes(include=[np.number]).columns:
                try:
                    corr = X_data[col].corr(y_true)
                    correlations[col] = float(corr)
                except:
                    correlations[col] = 0.0

            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            context["feature_correlations"] = dict(sorted_correlations[:10])

        return context

    def _generate_single_insight(self, context: Dict[str, Any], insight_type: str) -> str:
        """Generate a single insight using configured LLM provider"""

        # Create prompt based on insight type
        prompts = {
            "prediction_interpretation": f"""
            Based on the following ML model predictions and context, provide a human-like interpretation of what the predictions mean for business:

            Dataset Type: {context['dataset_type']}
            Target Column: {context['target_column']}
            Number of Samples: {context['num_samples']}

            Prediction Statistics:
            - Mean: {context.get('predictions', {}).get('mean', 'N/A')}
            - Range: {context.get('predictions', {}).get('min', 'N/A')} to {context.get('predictions', {}).get('max', 'N/A')}

            Please format your response with:
            1. A brief 1-2 sentence summary of prediction interpretation
            2. ## Key Findings section with bullet points
            3. Use **bold** for important metrics and values
            4. Include 2-3 main insights about what these predictions indicate
            """,

            "feature_analysis": f"""
            Analyze the top predictive features and explain their business impact:

            Dataset Type: {context['dataset_type']}
            Target Column: {context['target_column']}

            Top Features:
            {json.dumps(context.get('top_features', []), indent=2)}

            Feature Correlations:
            {json.dumps(context.get('feature_correlations', {}), indent=2)}

            Please format your response with:
            1. A headline: ## Feature Importance Analysis
            2. Describe which features are most important using **bold** for feature names
            3. Include bullet points for why each top feature matters
            4. Add a ## Business Impact section with implications
            5. Use specific correlation values from the data
            """,

            "trend_comparison": f"""
            Compare the predictions with historical patterns and identify trends:

            Dataset Type: {context['dataset_type']}
            Target Column: {context['target_column']}

            Current Statistics:
            - Predictions Mean: {context.get('predictions', {}).get('mean', 'N/A')}
            - True Values Mean: {context.get('true_values', {}).get('mean', 'N/A')}

            Top Features: {', '.join([f['feature'] for f in context.get('top_features', [])[:3]])}

            Please format your response with:
            1. A concise summary of trends identified
            2. ## Historical Patterns section
            3. Use bullet points to list key trends
            4. Bold important metrics like **mean**, **range**, and feature names
            5. Add ## Future Outlook section with implications
            """,

            "actionable_recommendations": f"""
            Generate specific, actionable business recommendations based on the model insights:

            Dataset Type: {context['dataset_type']}
            Target Column: {context['target_column']}

            Key Insights:
            - Top Feature: {context.get('top_features', [{}])[0].get('feature', 'N/A')}
            - Prediction Range: {context.get('predictions', {}).get('min', 'N/A')} - {context.get('predictions', {}).get('max', 'N/A')}

            Please format your response with:
            1. A brief introduction
            2. ## Recommended Actions section with 3-4 specific actions
            3. Format each action as a bullet point starting with action verbs
            4. Use **bold** for important metrics, features, or actions
            5. Add ## Expected Outcomes section describing benefits
            6. Keep recommendations practical and data-driven
            """
        }

        prompt = prompts.get(insight_type, "Provide general business insights based on the ML model predictions.")
        prompt += "\n\nAdditional requirements: Always use headers (## Section Name), bullet points with dashes, and **bold text** for important values."

        last_error = None
        for model_name in self.llm_models:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert business analyst and data scientist providing actionable insights from machine learning models. Your responses should be professional, concise, and focused on business value."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=450,
                    temperature=0.7
                )

                content = response.choices[0].message.content
                if content:
                    return content.strip()
                return f"Model '{model_name}' returned an empty response for {insight_type.replace('_', ' ')} insight."
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM model '{model_name}' failed: {last_error}")

                # Hard stop on auth/quota issues; fallback models will not help.
                normalized = last_error.lower()
                if (
                    "invalid api key" in normalized
                    or "incorrect api key" in normalized
                    or "authentication" in normalized
                    or "insufficient_quota" in normalized
                    or "quota" in normalized
                    or "billing" in normalized
                ):
                    break

        if last_error:
            return (
                f"Unable to generate {insight_type.replace('_', ' ')} insight due to API error: "
                f"{last_error}"
            )

        return f"Unable to generate {insight_type.replace('_', ' ')} insight due to API error."

    def _summarize_shap_values(self, shap_values: np.ndarray, X_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize SHAP values for reporting"""
        return {
            "mean_absolute_shap": float(np.abs(shap_values).mean()),
            "max_shap_value": float(np.max(np.abs(shap_values))),
            "shap_value_range": {
                "min": float(np.min(shap_values)),
                "max": float(np.max(shap_values))
            },
            "feature_count": len(X_data.columns)
        }

    def _summarize_predictions(self, predictions: np.ndarray, y_true: Optional[pd.Series]) -> Dict[str, Any]:
        """Summarize prediction results"""
        summary = {
            "num_predictions": len(predictions),
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        }

        if y_true is not None:
            # Calculate basic accuracy metrics if possible
            try:
                if len(np.unique(y_true)) <= 10:  # Classification
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_true, predictions)
                    summary["accuracy"] = float(accuracy)
                    summary["problem_type"] = "classification"
                else:  # Regression
                    from sklearn.metrics import mean_squared_error, r2_score
                    mse = mean_squared_error(y_true, predictions)
                    r2 = r2_score(y_true, predictions)
                    summary["mse"] = float(mse)
                    summary["r2_score"] = float(r2)
                    summary["problem_type"] = "regression"
            except:
                pass

        return summary

    def generate_comprehensive_report(
        self,
        model: BaseEstimator,
        X_data: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        predictions: Optional[np.ndarray] = None,
        target_column: str = "",
        dataset_type: str = "General",
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive business intelligence report

        Args:
            model: Trained ML model
            X_data: Feature data
            y_true: True target values
            predictions: Model predictions
            target_column: Name of target column
            dataset_type: Type of dataset
            historical_data: Historical dataset for trend comparison

        Returns:
            Comprehensive report dictionary
        """

        # Get basic analysis
        analysis = self.analyze_predictions(
            model=model,
            X_data=X_data,
            y_true=y_true,
            predictions=predictions,
            target_column=target_column,
            dataset_type=dataset_type
        )

        if not analysis["success"]:
            return analysis

        # Add historical trend analysis if available
        if historical_data is not None:
            trend_analysis = self._analyze_historical_trends(
                historical_data=historical_data,
                current_predictions=predictions,
                target_column=target_column
            )
            analysis["trend_analysis"] = trend_analysis

        # Add risk assessment
        risk_assessment = self._assess_business_risks(
            feature_importance=analysis["feature_importance"],
            predictions=predictions,
            dataset_type=dataset_type
        )
        analysis["risk_assessment"] = risk_assessment

        # Add implementation roadmap
        roadmap = self._generate_implementation_roadmap(
            insights=analysis["insights"],
            dataset_type=dataset_type
        )
        analysis["implementation_roadmap"] = roadmap

        return analysis

    def _analyze_historical_trends(
        self,
        historical_data: pd.DataFrame,
        current_predictions: np.ndarray,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze historical trends compared to current predictions"""

        # This is a simplified implementation - in practice, you'd do more sophisticated trend analysis
        trends = {
            "historical_avg": float(historical_data[target_column].mean()) if target_column in historical_data.columns else None,
            "current_avg": float(np.mean(current_predictions)),
            "trend_direction": "stable",  # Would be calculated based on time series analysis
            "insights": []
        }

        if trends["historical_avg"] and trends["current_avg"]:
            diff = ((trends["current_avg"] - trends["historical_avg"]) / trends["historical_avg"]) * 100
            if abs(diff) > 10:
                trends["trend_direction"] = "increasing" if diff > 0 else "decreasing"
                trends["insights"].append(".1f")

        return trends

    def _assess_business_risks(
        self,
        feature_importance: List[Dict[str, Any]],
        predictions: np.ndarray,
        dataset_type: str
    ) -> Dict[str, Any]:
        """Assess business risks based on model insights"""

        # Identify high-risk scenarios
        risk_factors = []

        # Check for high variability in predictions
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)

        if pred_std / pred_mean > 0.5:  # High variability
            risk_factors.append("High prediction variability indicates unstable business conditions")

        # Check for over-reliance on few features
        if len(feature_importance) > 0:
            top_feature_importance = feature_importance[0]["importance"]
            total_importance = sum(f["importance"] for f in feature_importance)

            if top_feature_importance / total_importance > 0.5:
                risk_factors.append(f"Over-reliance on '{feature_importance[0]['feature']}' feature - single point of failure")

        return {
            "risk_level": "High" if len(risk_factors) > 2 else "Medium" if len(risk_factors) > 0 else "Low",
            "risk_factors": risk_factors,
            "mitigation_suggestions": [
                "Diversify data sources to reduce single-point dependencies",
                "Implement monitoring for prediction stability",
                "Regular model retraining with new data"
            ]
        }

    def _generate_implementation_roadmap(
        self,
        insights: List[Dict[str, Any]],
        dataset_type: str
    ) -> List[Dict[str, Any]]:
        """Generate an implementation roadmap based on insights"""

        roadmap = [
            {
                "phase": "Immediate Actions (Week 1)",
                "actions": [
                    "Review top predictive features and their business impact",
                    "Identify quick wins from actionable recommendations",
                    "Set up monitoring for key metrics"
                ],
                "timeline": "1 week",
                "resources_needed": ["Data team", "Business stakeholders"]
            },
            {
                "phase": "Short-term Implementation (1-3 months)",
                "actions": [
                    "Implement recommended changes based on model insights",
                    "Set up automated reporting for predictions",
                    "Train team on using AI insights for decision making"
                ],
                "timeline": "1-3 months",
                "resources_needed": ["IT team", "Operations team", "Training budget"]
            },
            {
                "phase": "Long-term Optimization (3-6 months)",
                "actions": [
                    "Integrate AI insights into business processes",
                    "Develop custom dashboards for real-time monitoring",
                    "Establish feedback loop for continuous model improvement"
                ],
                "timeline": "3-6 months",
                "resources_needed": ["Development team", "Business intelligence tools"]
            }
        ]

        return roadmap