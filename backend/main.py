from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import json
import pickle
from typing import Dict, Any
import os
from datetime import datetime
from pandas.api import types as pd_types
import logging

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import numpy as np

# AI Advisor import
from ai_advisor import AIAdvisor
from rag_chatbot import BusinessAnalyticsRAG
from what_if_simulator import simulate_scenario, compare_results, generate_explanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Store uploaded files temporarily
UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Global variables to store dataframes for cleaning
current_dataframe = None
current_filename = None
cleaned_dataframe = None
current_fill_methods = None

# ML training variables
trained_models = None
ml_results = None
best_model = None
training_artifacts = {}

# AI Advisor
ai_advisor = None

# RAG chatbot
rag_chatbot = None
latest_ai_insights_report = {}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Data Analysis API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        global current_dataframe, current_filename
        
        # Read the uploaded file
        contents = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"success": False, "error": "Unsupported file format. Please upload CSV or Excel file."}
        
        # Store the dataframe for later cleaning
        current_dataframe = df.copy()
        current_filename = file.filename
        
        # Calculate dataset statistics
        stats = calculate_dataset_stats(df)
        stats["filename"] = file.filename
        stats["success"] = True
        
        return stats
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive statistics about the dataset"""
    
    # Basic info
    num_rows = len(df)
    num_columns = len(df.columns)
    columns_list = df.columns.tolist()
    
    # Missing values
    missing_values = df.isnull().sum().astype(int).to_dict()
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / (num_rows * num_columns) * 100) if (num_rows * num_columns) > 0 else 0
    
    # Duplicate rows
    num_duplicates = df.duplicated().sum()
    
    # Data types
    data_types = df.dtypes.astype(str).to_dict()
    
    # Summary statistics for numeric columns
    numeric_summary = {}
    for col in df.select_dtypes(include=['number']).columns:
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Handle NaN values
        numeric_summary[col] = {
            "mean": float(mean_val) if pd.notna(mean_val) else None,
            "median": float(median_val) if pd.notna(median_val) else None,
            "std": float(std_val) if pd.notna(std_val) else None,
            "min": float(min_val) if pd.notna(min_val) else None,
            "max": float(max_val) if pd.notna(max_val) else None,
            "missing": int(df[col].isnull().sum())
        }
    
    # Categorical columns summary
    categorical_summary = {}
    for col in df.select_dtypes(include=['object']).columns:
        categorical_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "most_common": str(df[col].value_counts().index[0]) if len(df[col].value_counts()) > 0 else None,
            "missing": int(df[col].isnull().sum())
        }
    
    return {
        "success": True,
        "num_rows": num_rows,
        "num_columns": num_columns,
        "columns": columns_list,
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing": int(total_missing),
        "missing_percentage": round(missing_percentage, 2),
        "num_duplicates": int(num_duplicates),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary
    }

@app.post("/cleaning-suggestions")
async def get_cleaning_suggestions() -> Dict[str, Any]:
    """Get AI-powered cleaning suggestions based on dataset analysis"""
    try:
        global current_dataframe
        
        if current_dataframe is None:
            return {"success": False, "error": "No dataset loaded. Please upload a file first."}
        
        df = current_dataframe.copy()
        suggestions = {
            "removeDuplicates": False,
            "deleteHighMissing": False,
            "deleteLowMissing": False,
            "fillMissing": False
        }
        explanations = []
        fill_methods = {}
        
        # 1. Check for duplicate rows
        num_duplicates = df.duplicated().sum()
        if num_duplicates > 0:
            suggestions["removeDuplicates"] = True
            duplicate_percentage = (num_duplicates / len(df)) * 100
            explanations.append(f"🔄 Found {num_duplicates} duplicate rows ({duplicate_percentage:.1f}%) - recommend removing them")
        
        # 2. Check for columns with >40% missing data
        high_missing_cols = []
        for col in df.columns:
            missing_percent = (df[col].isnull().sum() / len(df)) * 100
            if missing_percent > 40:
                high_missing_cols.append(f"{col} ({missing_percent:.1f}%)")
        
        if high_missing_cols:
            suggestions["deleteHighMissing"] = True
            explanations.append(f"🗑️ {len(high_missing_cols)} columns have >40% missing values: {', '.join(high_missing_cols[:3])}{'...' if len(high_missing_cols) > 3 else ''} - recommend deleting these columns")
        
        # 3. Check percentage of rows with missing values for row deletion vs filling
        # Calculate how many rows have at least one missing value
        rows_with_missing = df.isnull().any(axis=1).sum()
        rows_with_missing_percentage = (rows_with_missing / len(df)) * 100 if len(df) > 0 else 0
        total_missing = df.isnull().sum().sum()
        
        if rows_with_missing_percentage < 5 and rows_with_missing > 0:
            # Less than 5% of rows have missing values - suggest removing those rows
            suggestions["deleteLowMissing"] = True
            explanations.append(f"📊 {rows_with_missing} rows have missing values ({rows_with_missing_percentage:.1f}% of rows, <5%) - recommend removing these rows")
        elif total_missing > 0:
            # 5% or more of rows have missing values - suggest filling instead
            suggestions["fillMissing"] = True
            explanations.append(f"✅ {rows_with_missing} rows have missing values ({rows_with_missing_percentage:.1f}% of rows, ≥5%) - recommend filling missing values")
        
        # 4. Determine filling methods for each column with missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_percent = (df[col].isnull().sum() / len(df)) * 100
                
                # Skip if column will be deleted
                if missing_percent > 40:
                    continue
                
                # Determine data type
                try:
                    is_numeric = pd_types.is_numeric_dtype(df[col])
                except Exception:
                    is_numeric = False
                
                if not is_numeric:
                    # Categorical - use mode
                    fill_methods[col] = "mode"
                else:
                    # Numeric - check for outliers/high variation
                    # Calculate IQR to detect outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Check if data has high variation (many outliers or high std relative to mean)
                    std_val = df[col].std()
                    mean_val = df[col].mean()
                    
                    # High variation indicators:
                    # 1. Std > |mean| * 0.5 (high relative variation)
                    # 2. Many outliers (values beyond 1.5*IQR)
                    has_high_variation = False
                    
                    if pd.notna(std_val) and pd.notna(mean_val) and abs(mean_val) > 0:
                        relative_std = std_val / abs(mean_val)
                        if relative_std > 0.5:  # High relative variation
                            has_high_variation = True
                    
                    # Count outliers
                    if pd.notna(IQR) and IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        outlier_percentage = (outlier_count / len(df)) * 100
                        if outlier_percentage > 10:  # More than 10% outliers
                            has_high_variation = True
                    
                    if has_high_variation:
                        fill_methods[col] = "median"
                        explanations.append(f"📈 Column '{col}' has high variation/outliers - recommend filling with median")
                    else:
                        fill_methods[col] = "mean"
                        explanations.append(f"📊 Column '{col}' has normal variation - recommend filling with mean")
        
        # Overall assessment
        enabled_options = [k for k, v in suggestions.items() if v]
        if not enabled_options:
            explanations.append("🎉 Dataset looks clean! All options are optional.")
        else:
            explanations.append(f"🤖 AI recommends: {', '.join(enabled_options)}")
        
        # Store fill methods for use in cleaning
        global current_fill_methods
        current_fill_methods = fill_methods
        
        return {
            "success": True,
            "suggestions": suggestions,
            "explanations": explanations,
            "fill_methods": fill_methods,
            "dataset_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "duplicates": int(num_duplicates),
                "rows_with_missing": int(rows_with_missing),
                "rows_with_missing_percentage": float(rows_with_missing_percentage),
                "missing_values": int(total_missing),
                "high_missing_cols": len(high_missing_cols)
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/clean")
async def clean_data(cleaning_request: Dict[str, Any]) -> Dict[str, Any]:
    """Clean the dataset based on specified rules"""
    try:
        global current_dataframe
        
        if current_dataframe is None:
            return {"success": False, "error": "No dataset loaded. Please upload a file first."}
        
        df = current_dataframe.copy()
        original_rows = len(df)
        original_cols = len(df.columns)
        changes_made = []
        
        cleaning_config = cleaning_request.get('cleaning_config', {})
        
        # 1. Remove duplicates
        if cleaning_config.get('removeDuplicates', True):
            duplicates_count = df.duplicated().sum()
            if duplicates_count > 0:
                df = df.drop_duplicates()
                changes_made.append(f"🗑️ Removed {duplicates_count} duplicate rows")
        
        # 2. Delete columns with > 40% missing values
        if cleaning_config.get('deleteHighMissing', True):
            cols_to_delete = []
            for col in df.columns:
                missing_percent = (df[col].isnull().sum() / len(df)) if len(df) > 0 else 0
                if missing_percent > 0.40:
                    cols_to_delete.append(col)
            
            if cols_to_delete:
                df = df.drop(columns=cols_to_delete)
                changes_made.append(f"🗑️ Deleted {len(cols_to_delete)} columns with >40% missing values: {', '.join(cols_to_delete)}")
        
        # 3. Delete rows with > 5% missing values per row
        if cleaning_config.get('deleteLowMissing', True):
            total_cols = len(df.columns)
            rows_to_delete = []
            for idx, row in df.iterrows():
                missing_count = row.isnull().sum()
                missing_percent = (missing_count / total_cols) if total_cols > 0 else 0
                if missing_percent > 0.05:
                    rows_to_delete.append(idx)
            
            if rows_to_delete:
                df = df.drop(rows_to_delete)
                changes_made.append(f"📊 Deleted {len(rows_to_delete)} rows with <5% missing values")
        
# 4. Fill remaining missing values with appropriate methods
        if cleaning_config.get('fillMissing', True):
            global current_fill_methods
            fill_count = 0
            for col in df.columns:
                if df[col].isnull().any():
                    # Count missing values BEFORE filling
                    missing_before = int(df[col].isnull().sum())
                    
                    # Determine fill method
                    fill_method = "median"  # default
                    if current_fill_methods and col in current_fill_methods:
                        fill_method = current_fill_methods[col]
                    
                    # Detect numeric dtype robustly
                    try:
                        is_numeric = pd_types.is_numeric_dtype(df[col])
                    except Exception:
                        is_numeric = False
                    
                    if is_numeric:
                        if fill_method == "mean":
                            mean_val = df[col].mean()
                            if pd.isna(mean_val):
                                mean_val = 0
                            df[col] = df[col].fillna(mean_val)
                        elif fill_method == "median":
                            median_val = df[col].median()
                            if pd.isna(median_val):
                                median_val = 0
                            df[col] = df[col].fillna(median_val)
                        else:
                            # fallback to median
                            median_val = df[col].median()
                            if pd.isna(median_val):
                                median_val = 0
                            df[col] = df[col].fillna(median_val)
                    else:
                        # Categorical - use mode
                        try:
                            mode_val = df[col].mode()
                        except Exception:
                            mode_val = []
                        
                        if hasattr(mode_val, '__len__') and len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
                        else:
                            df[col] = df[col].fillna('Unknown')
                    
                    # Add the count of values that were filled
                    fill_count += missing_before
            
            if fill_count > 0:
                changes_made.append(f"✅ Filled {fill_count} missing values using appropriate methods (mean/median/mode)")

        # Ensure no duplicate rows remain after fills
        duplicates_after = int(df.duplicated().sum())
        if duplicates_after > 0:
            df = df.drop_duplicates()
            changes_made.append(f"🗑️ Removed {duplicates_after} duplicate rows after filling")

        # Final fallback: if any missing values still remain, force-fill with safe defaults
        remaining_missing = int(df.isnull().sum().sum())
        if remaining_missing > 0:
            forced_fill = 0
            for col in df.columns:
                miss = int(df[col].isnull().sum())
                if miss > 0:
                    try:
                        is_numeric = pd_types.is_numeric_dtype(df[col])
                    except Exception:
                        is_numeric = False

                    if is_numeric:
                        median_val = df[col].median()
                        if pd.isna(median_val):
                            median_val = 0
                        df[col] = df[col].fillna(median_val)
                    else:
                        mode_val = df[col].mode()
                        if hasattr(mode_val, '__len__') and len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
                        else:
                            df[col] = df[col].fillna('Unknown')

                    forced_fill += miss

            if forced_fill > 0:
                changes_made.append(f"⚠️ Forced fill of {forced_fill} remaining missing values using fallback defaults")
        
        # Calculate cleaned stats
        cleaned_stats = calculate_dataset_stats(df)
        cleaned_stats["filename"] = current_filename
        
        # Get sample data (first 7 rows)
        sample_df = df.head(7)
        sample_data = sample_df.to_dict('records')
        
        # Store cleaned dataframe for download
        global cleaned_dataframe
        cleaned_dataframe = df
        
        return {
            "success": True,
            "original_rows": original_rows,
            "original_cols": original_cols,
            "cleaned_stats": cleaned_stats,
            "changes_made": changes_made,
            "sample_data": sample_data
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/download-cleaned")
async def download_cleaned():
    """Download the cleaned dataset as CSV"""
    try:
        global cleaned_dataframe, current_filename
        
        if cleaned_dataframe is None:
            return {"error": "No cleaned data available"}
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        cleaned_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        filename = f"cleaned_{current_filename}" if current_filename else "cleaned_data.csv"
        
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        return {"error": str(e)}

@app.post("/train-ml")
async def train_ml_models(request: Dict[str, Any]) -> Dict[str, Any]:
    """Train multiple ML models and select the best one"""
    try:
        global cleaned_dataframe, trained_models, ml_results, best_model, training_artifacts
        
        if cleaned_dataframe is None:
            # Try to use current_dataframe as fallback if no cleaning was done
            if current_dataframe is not None:
                cleaned_dataframe = current_dataframe.copy()
                print("ML Training: Using original dataframe as cleaned_dataframe was None")
            else:
                return {"success": False, "error": "No dataset available. Please upload and clean your data first."}
        
        df = cleaned_dataframe.copy()
        
        # Check if we have enough data for training
        if len(df) < 10:
            return {"success": False, "error": "Dataset must have at least 10 rows for ML training"}
        
        target_column = request.get('target_column')
        if not target_column:
            return {"success": False, "error": "Target column must be specified"}
        
        if target_column not in df.columns:
            return {"success": False, "error": f"Target column '{target_column}' not found in dataset"}
        
        # Prepare data for ML
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if len(X.columns) < 1:
            return {"success": False, "error": "Dataset must have at least 1 feature column (excluding target)"}
        
        # Determine if it's a classification or regression problem
        is_classification = False
        try:
            pd_types.is_numeric_dtype(y)
            # Check if target has few unique values (likely classification)
            unique_values = y.nunique()
            if unique_values <= 10:  # Arbitrary threshold
                is_classification = True
        except:
            is_classification = True
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.columns:
            if not pd_types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Encode target if classification
        target_encoder = None
        if is_classification and not pd_types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {}
        results = {}
        
        if is_classification:
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'XGBoost Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'SVM': SVC(random_state=42, kernel='rbf'),
                'Naive Bayes': GaussianNB()
            }
            
            # Train and evaluate each model
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    
                    results[name] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'cv_mean': float(cv_scores.mean()),
                        'cv_std': float(cv_scores.std()),
                        'model_type': 'classification'
                    }
                    
                except Exception as e:
                    results[name] = {'error': str(e), 'model_type': 'classification'}
        
        else:  # Regression
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'XGBoost Regressor': XGBRegressor(random_state=42)
            }
            
            # Train and evaluate each model
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    
                    results[name] = {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r2_score': float(r2),
                        'cv_rmse': float(cv_rmse),
                        'cv_std': float(cv_scores.std()),
                        'model_type': 'regression'
                    }
                    
                except Exception as e:
                    results[name] = {'error': str(e), 'model_type': 'regression'}
        
        # Find best model
        best_model_name = None
        best_score = -float('inf') if is_classification else float('inf')
        
        for name, metrics in results.items():
            if 'error' in metrics:
                continue
                
            if is_classification:
                score = metrics['f1_score']  # Use F1-score for classification
            else:
                score = metrics['rmse']  # Use RMSE for regression (lower is better)
                
            if (is_classification and score > best_score) or (not is_classification and score < best_score):
                best_score = score
                best_model_name = name
        
        # Store results
        trained_models = models
        ml_results = results
        best_model = best_model_name
        training_artifacts = {
            "feature_columns": list(X.columns),
            "label_encoders": label_encoders,
            "target_encoder": target_encoder,
            "scaler": scaler,
            "is_classification": is_classification,
            "target_column": target_column,
            "feature_types": {
                col: (
                    "categorical" if col in label_encoders else
                    "numeric" if pd_types.is_numeric_dtype(df[col]) else
                    "other"
                )
                for col in X.columns
            },
            "category_options": {
                col: [str(v) for v in encoder.classes_.tolist()]
                for col, encoder in label_encoders.items()
            },
            "feature_defaults": {
                col: (
                    float(df[col].median())
                    if pd_types.is_numeric_dtype(df[col]) and pd.notna(df[col].median())
                    else (
                        str(df[col].mode().iloc[0])
                        if len(df[col].mode()) > 0
                        else ""
                    )
                )
                for col in X.columns
            }
        }
        
        return {
            "success": True,
            "problem_type": "classification" if is_classification else "regression",
            "target_column": target_column,
            "results": results,
            "best_model": best_model_name,
            "dataset_info": {
                "total_samples": len(df),
                "features": len(X.columns),
                "feature_columns": list(X.columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/download-best-model")
async def download_best_model():
    """Download the trained best model and preprocessing artifacts as a pickle bundle."""
    try:
        global trained_models, best_model, training_artifacts

        if trained_models is None or best_model is None:
            return {"success": False, "error": "No trained best model available. Please train models first."}

        model_obj = trained_models.get(best_model)
        if model_obj is None:
            return {"success": False, "error": f"Best model '{best_model}' is not available."}

        safe_model_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(best_model)).strip("_")
        if not safe_model_name:
            safe_model_name = "best_model"

        bundle = {
            "model_name": best_model,
            "model": model_obj,
            "training_artifacts": training_artifacts or {},
            "exported_at": datetime.now().isoformat(),
        }

        buffer = io.BytesIO()
        pickle.dump(bundle, buffer)
        buffer.seek(0)

        filename = f"{safe_model_name}.pkl"
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.exception("Error exporting best model")
        return {"success": False, "error": f"Error exporting best model: {str(e)}"}

def detect_dataset_type(df: pd.DataFrame, target_column: str, X: pd.DataFrame) -> tuple:
    """
    Detect the type of dataset (Sales, HR, Customer, Finance, etc.)
    Returns: (dataset_type, confidence_score)
    """
    
    column_names = [col.lower() for col in df.columns]
    target_lower = target_column.lower()
    
    # Keywords for different dataset types
    sales_keywords = ['price', 'sales', 'revenue', 'quantity', 'order', 'customer_id', 'product', 'discount', 'profit', 'cost']
    hr_keywords = ['salary', 'employee', 'department', 'experience', 'age', 'performance', 'attrition', 'job_role', 'satisfaction']
    finance_keywords = ['revenue', 'expense', 'profit', 'investment', 'return', 'risk', 'portfolio', 'market', 'stock', 'interest']
    customer_keywords = ['customer', 'churn', 'purchase', 'satisfaction', 'loyalty', 'subscription', 'retention', 'complaint']
    healthcare_keywords = ['patient', 'disease', 'treatment', 'medication', 'diagnosis', 'health', 'mortality', 'readmission']
    marketing_keywords = ['campaign', 'conversion', 'engagement', 'impression', 'click', 'ctr', 'roi', 'reach', 'audience']
    production_keywords = ['production', 'quality', 'defect', 'efficiency', 'yield', 'throughput', 'downtime', 'maintenance']
    
    dataset_types = {
        'Sales': sales_keywords,
        'HR / Human Resources': hr_keywords,
        'Finance / Investment': finance_keywords,
        'Customer Analytics': customer_keywords,
        'Healthcare': healthcare_keywords,
        'Marketing': marketing_keywords,
        'Production / Manufacturing': production_keywords
    }
    
    # Count keyword matches
    matches = {}
    for dtype, keywords in dataset_types.items():
        count = sum(1 for col in column_names for kw in keywords if kw in col)
        if target_lower and any(kw in target_lower for kw in keywords):
            count += 2  # Boost if target matches
        matches[dtype] = count
    
    # Get best match
    best_match = max(matches, key=matches.get)
    confidence = matches[best_match] / (len(df.columns) + 2) if matches[best_match] > 0 else 0.3
    confidence = min(confidence, 1.0)
    
    if matches[best_match] == 0:
        return 'General / Other', 0.3
    
    return best_match, confidence

def generate_business_recommendations(dataset_type: str, target_column: str, df: pd.DataFrame, 
                                    X: pd.DataFrame, y: pd.Series, feature_importance: list,
                                    problem_type: str) -> list:
    """Generate business-specific recommendations based on dataset type"""
    
    recommendations = []
    
    if 'Sales' in dataset_type:
        # Sales-specific recommendations
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        for feat in feature_importance[:5]:
            feature_name = feat['feature']
            try:
                corr = df[feature_name].corr(y) if feature_name in df.columns else 0
                
                if 'price' in feature_name.lower() and corr < -0.3:
                    recommendations.append({
                        "title": f"Optimize {feature_name} Strategy",
                        "action": f"Implement dynamic pricing or targeted discounts for high-impact products",
                        "expected_outcome": f"Since lower '{feature_name}' correlates with higher {target_column}, consider strategic price reductions of 10-25% for key products",
                        "business_impact": "Increased sales volume, market share growth, and improved customer acquisition"
                    })
                elif 'discount' in feature_name.lower() or 'promotion' in feature_name.lower():
                    if corr > 0.3:
                        recommendations.append({
                            "title": f"Enhance Promotional Campaigns",
                            "action": f"Allocate more budget to promotional activities",
                            "expected_outcome": f"Positive correlation with {target_column} suggests increased promotions yield better results",
                            "business_impact": "Higher conversion rates, increased average order value, and improved ROI on marketing spend"
                        })
                elif 'quantity' in feature_name.lower():
                    if corr > 0.4:
                        recommendations.append({
                            "title": f"Focus on Bulk Sales & Bundles",
                            "action": f"Create bundle offers and incentivize bulk purchases",
                            "expected_outcome": f"Strong correlation suggests customers buying in larger quantities significantly boost {target_column}",
                            "business_impact": "Reduced overhead per unit, improved inventory turnover, and better profit margins"
                        })
            except:
                pass
        
        if len(recommendations) == 0:
            recommendations.append({
                "title": "Price Optimization",
                "action": "Conduct A/B testing with different price points",
                "expected_outcome": f"Identify optimal price range that maximizes {target_column} while maintaining profitability",
                "business_impact": "Improved revenue, market competitiveness, and customer satisfaction"
            })
    
    elif 'HR' in dataset_type:
        # HR-specific recommendations
        for feat in feature_importance[:3]:
            feature_name = feat['feature']
            try:
                corr = df[feature_name].corr(y) if feature_name in df.columns else 0
                
                if 'salary' in feature_name.lower() and corr < -0.2:
                    recommendations.append({
                        "title": "Salary Structure Review",
                        "action": "Benchmark salaries against industry standards and adjust compensation",
                        "expected_outcome": f"Competitive salaries correlated with better {target_column}",
                        "business_impact": "Improved employee retention, higher productivity, and reduced turnover costs"
                    })
                elif 'experience' in feature_name.lower():
                    recommendations.append({
                        "title": "Invest in Employee Development",
                        "action": f"Create training and mentorship programs to develop employee skills",
                        "expected_outcome": f"Experience directly impacts {target_column} - skilled employees drive better results",
                        "business_impact": "Enhanced team capabilities, better project outcomes, and employee career growth"
                    })
            except:
                pass
        
        if len(recommendations) == 0:
            recommendations.append({
                "title": "Employee Performance Management",
                "action": "Implement regular feedback and performance evaluation systems",
                "expected_outcome": f"Data-driven insights help identify and address performance gaps",
                "business_impact": "Improved overall employee performance and organizational productivity"
            })
    
    elif 'Finance' in dataset_type:
        # Finance-specific recommendations
        recommendations.append({
            "title": "Portfolio Optimization",
            "action": "Rebalance investments based on model predictions",
            "expected_outcome": f"Allocate resources to high-impact factors identified by the model",
            "business_impact": "Improved returns, reduced risk, and better capital allocation"
        })
        
        recommendations.append({
            "title": "Risk Management",
            "action": "Monitor key financial indicators highlighted by feature importance",
            "expected_outcome": "Early identification of risks and market opportunities",
            "business_impact": "Minimized losses, protected investments, and maximized ROI"
        })
    
    elif 'Customer' in dataset_type:
        # Customer-specific recommendations
        recommendations.append({
            "title": "Customer Retention Program",
            "action": "Identify at-risk customers and implement retention strategies",
            "expected_outcome": f"Proactively address factors that impact {target_column}",
            "business_impact": "Reduced churn, increased lifetime value, and improved loyalty"
        })
        
        recommendations.append({
            "title": "Personalized Marketing",
            "action": "Segment customers based on model predictions and customize offers",
            "expected_outcome": "Targeted campaigns with higher conversion rates",
            "business_impact": "Increased customer satisfaction and higher marketing ROI"
        })
    
    else:
        # Generic recommendations
        recommendations.append({
            "title": "Focus on Top Predictive Factors",
            "action": f"Monitor and optimize the {len(feature_importance)} key factors identified",
            "expected_outcome": f"Prioritize efforts on high-impact factors affecting {target_column}",
            "business_impact": "Improved efficiency, better resource allocation, and stronger results"
        })
    
    return recommendations[:5]  # Return top 5 recommendations

def generate_business_insights(dataset_type: str, target_column: str, df: pd.DataFrame, 
                              problem_type: str) -> list:
    """Generate business insights based on dataset type"""
    
    insights = []
    
    if 'Sales' in dataset_type:
        insights.append("📊 Sales Insight: Focus on customer segments with highest purchase frequency and order value")
        insights.append("💰 Revenue Opportunity: Seasonal patterns can be leveraged for targeted marketing campaigns")
        insights.append("🎯 Strategic Action: Bundle popular products with slower-moving items to increase overall basket size")
    
    elif 'HR' in dataset_type:
        insights.append("👥 HR Insight: Monitor early warning signs of employee disengagement to reduce turnover")
        insights.append("💼 Talent Strategy: Invest in high-potential employees identified through performance metrics")
        insights.append("🎓 Development: Create targeted training programs based on skill gaps identified in the data")
    
    elif 'Finance' in dataset_type:
        insights.append("📈 Market Insight: Track correlations between economic indicators and financial performance")
        insights.append("💵 Investment Strategy: Use predictions to optimize portfolio allocation and reduce risk exposure")
        insights.append("📊 Analysis: Regular monitoring of key financial ratios helps maintain fiscal health")
    
    elif 'Customer' in dataset_type:
        insights.append("🔄 Churn Prevention: Early identification of at-risk customers enables proactive intervention")
        insights.append("💎 VIP Strategy: Identify and nurture high-value customers with exclusive benefits")
        insights.append("🛍️ Engagement: Personalized interactions based on customer behavior increase satisfaction")
    
    else:
        total_samples = len(df)
        if total_samples > 10000:
            insights.append(f"📊 Large Dataset: Your {total_samples} records provide substantial data for reliable predictions")
        insights.append(f"🎯 Target Focus: {target_column} is the primary metric to monitor and improve")
        insights.append("📈 Continuous Improvement: Regularly validate predictions against actual outcomes")
    
    return insights

@app.post("/get-ai-suggestions")
async def get_ai_suggestions(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI suggestions and insights about the model"""
    try:
        global cleaned_dataframe, trained_models, X_train, X_test, y_train, y_test, best_model, is_classification
        
        if cleaned_dataframe is None or trained_models is None:
            return {"success": False, "error": "No trained model available. Please train a model first."}
        
        df = cleaned_dataframe.copy()
        target_column = request.get('target_column')
        best_model_name = request.get('best_model')
        problem_type = request.get('problem_type', 'classification')
        
        # Get model results
        ml_results_dict = request.get('results', {})
        dataset_info = request.get('dataset_info', {})
        
        # ============================================================================
        # PART 1: ANALYZE FEATURE IMPORTANCE
        # ============================================================================
        
        # Get feature importance based on correlations or model feature importance
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        feature_importance_list = []
        
        # Try to get feature importance from model
        best_model_obj = trained_models.get(best_model_name)
        
        if best_model_obj and hasattr(best_model_obj, 'feature_importances_'):
            # Tree-based model with feature_importances_
            importances = best_model_obj.feature_importances_
            feature_names = list(X.columns)
            
            for feature, importance in zip(feature_names, importances):
                feature_importance_list.append({
                    "feature": feature,
                    "importance": float(importance)
                })
        
        elif best_model_obj and hasattr(best_model_obj, 'coef_'):
            # Linear model with coefficients
            if len(best_model_obj.coef_.shape) > 1:
                # Multi-class classification
                coef = np.abs(best_model_obj.coef_[0])
            else:
                coef = np.abs(best_model_obj.coef_)
            
            feature_names = list(X.columns)
            total_coef = np.sum(np.abs(coef))
            
            for feature, importance in zip(feature_names, coef):
                feature_importance_list.append({
                    "feature": feature,
                    "importance": float(importance / total_coef) if total_coef > 0 else 0.0
                })
        else:
            # Use correlation with target for feature importance
            numeric_features = X.select_dtypes(include=[np.number]).columns
            correlations = []
            
            for feature in numeric_features:
                try:
                    corr = abs(df[feature].corr(y))
                    correlations.append((feature, float(corr)))
                except:
                    pass
            
            # Normalize correlations to [0, 1]
            if correlations:
                correlations = sorted(correlations, key=lambda x: x[1], reverse=True)
                max_corr = max(corr[1] for corr in correlations)
                
                for feature, corr in correlations:
                    feature_importance_list.append({
                        "feature": feature,
                        "importance": corr / max_corr if max_corr > 0 else 0.0
                    })
        
        # Sort by importance
        feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)
        
        # ============================================================================
        # PART 2: FEATURE IMPROVEMENT RECOMMENDATIONS
        # ============================================================================
        
        feature_recommendations = []
        
        # For numeric features, suggest whether to increase or decrease
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features[:10]:  # Top 10 features
            try:
                correlation = df[feature].corr(y)
                
                recommendation = {
                    "feature": feature,
                    "direction": "increase" if correlation > 0 else "decrease",
                    "reason": ""
                }
                
                if correlation > 0:
                    recommendation["reason"] = f"Positive correlation ({correlation:.3f}) with target: Increasing this value will help improve the '{target_column}' column."
                else:
                    recommendation["reason"] = f"Negative correlation ({correlation:.3f}) with target: Decreasing this value will help improve the '{target_column}' column."
                
                feature_recommendations.append(recommendation)
            except:
                pass
        
        # ============================================================================
        # PART 3: DATA SUGGESTIONS
        # ============================================================================
        
        data_suggestions = []
        
        # Check dataset size
        total_samples = dataset_info.get('total_samples', len(df))
        num_features = dataset_info.get('features', len(X.columns))
        
        if total_samples < 100:
            data_suggestions.append(f"Dataset is relatively small ({total_samples} samples). Consider collecting more data to improve model generalization. Ideally aim for at least 100-1000 samples depending on complexity.")
        elif total_samples < 500:
            data_suggestions.append(f"Current dataset has {total_samples} samples. Consider increasing to 500+ samples to better capture data patterns and improve model robustness.")
        elif total_samples > 100000:
            data_suggestions.append(f"Dataset is quite large ({total_samples} samples). Consider using a subset for faster training if computation time is an issue.")
        else:
            data_suggestions.append(f"Dataset size ({total_samples} samples) is good for training machine learning models.")
        
        # Check feature count
        if num_features < 3:
            data_suggestions.append(f"Limited number of features ({num_features}). Consider engineering new features or adding more relevant columns to improve model performance.")
        elif num_features > 50:
            data_suggestions.append(f"High number of features ({num_features}). Consider removing low-importance features to reduce model complexity and potential overfitting.")
        else:
            data_suggestions.append(f"Feature count ({num_features}) is appropriate for model training.")
        
        # Check for missing values
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 5:
            data_suggestions.append(f"Dataset has {missing_percentage:.1f}% missing values. Ensure all missing values are properly handled before using the model for critical decisions.")
        
        # ============================================================================
        # PART 4: MODEL SELECTION REASONING
        # ============================================================================
        
        model_selection_reasons = []
        best_model_metrics = ml_results_dict.get(best_model_name, {})
        
        if problem_type == 'classification':
            f1_score = best_model_metrics.get('f1_score', 0)
            accuracy = best_model_metrics.get('accuracy', 0)
            
            model_selection_reasons.append(f"Achieved the highest F1-Score ({f1_score:.4f}) among all tested models")
            model_selection_reasons.append(f"Best accuracy ({accuracy:.4f}) on the test dataset")
            
            # Compare with other models
            for model_name, metrics in ml_results_dict.items():
                if model_name != best_model_name and 'f1_score' in metrics:
                    if f1_score > metrics.get('f1_score', 0):
                        diff = ((f1_score - metrics.get('f1_score', 0)) / metrics.get('f1_score', 1)) * 100
                        model_selection_reasons.append(f"Outperformed {model_name} by {diff:.1f}% in F1-Score")
            
            model_selection_reasons.append("Best balance between Precision and Recall for accurate classification")
        else:
            rmse = best_model_metrics.get('rmse', 0)
            r2 = best_model_metrics.get('r2_score', 0)
            
            model_selection_reasons.append(f"Achieved the lowest RMSE ({rmse:.4f}) among all tested models")
            model_selection_reasons.append(f"Highest R² Score ({r2:.4f}) explaining prediction variance")
            
            # Compare with other models
            for model_name, metrics in ml_results_dict.items():
                if model_name != best_model_name and 'rmse' in metrics:
                    if rmse < metrics.get('rmse', float('inf')):
                        diff = ((metrics.get('rmse', 0) - rmse) / metrics.get('rmse', 1)) * 100
                        model_selection_reasons.append(f"More accurate than {model_name} by {diff:.1f}% (lower RMSE)")
            
            model_selection_reasons.append("Best prediction accuracy for continuous values")
        
        # ============================================================================
        # PART 5: KEY INSIGHTS
        # ============================================================================
        
        key_insights = []
        
        # Top feature insight
        if feature_importance_list:
            top_feature = feature_importance_list[0]
            key_insights.append(f"The feature '{top_feature['feature']}' has the most impact ({top_feature['importance']*100:.1f}%) on predictions.")
        
        # Dataset insight
        key_insights.append(f"Your dataset contains {num_features} features and {total_samples} samples, which is {('ideal' if 100 <= num_features <= 100 else 'good' if 3 <= num_features < 200 else 'consider optimizing')} for training.")
        
        # Model insight
        if problem_type == 'classification':
            best_metric = best_model_metrics.get('f1_score', 0)
            metric_name = "F1-Score"
        else:
            best_metric = best_model_metrics.get('r2_score', 0)
            metric_name = "R² Score"
        
        key_insights.append(f"Your {best_model_name} model achieved a {metric_name} of {best_metric:.4f}, indicating {'excellent' if best_metric > 0.9 else 'good' if best_metric > 0.7 else 'moderate' if best_metric > 0.5 else 'poor'} performance.")
        
        # ============================================================================
        # PART 6: PROBLEM TYPE ANALYSIS
        # ============================================================================
        
        problem_type_analysis = {
            "explanation": "",
            "reason": ""
        }
        
        if problem_type == 'classification':
            # Determine number of classes
            unique_target = y.nunique()
            if unique_target == 2:
                problem_type_analysis["explanation"] = f"This is a Binary Classification problem. You are predicting between {unique_target} possible outcomes for the '{target_column}' column."
                problem_type_analysis["reason"] = "The target column has only 2 unique values, making this a classic binary classification task."
            else:
                problem_type_analysis["explanation"] = f"This is a Multi-class Classification problem. You are predicting between {unique_target} possible categories for the '{target_column}' column."
                problem_type_analysis["reason"] = f"The target column has {unique_target} unique values, making this a multi-class classification problem."
        else:
            problem_type_analysis["explanation"] = "This is a Regression problem. You are predicting continuous numerical values for the '{target_column}' column."
            problem_type_analysis["reason"] = "The target column contains continuous numerical values requiring regression analysis."
        
        # ============================================================================
        # PART 7: GENERAL RECOMMENDATIONS
        # ============================================================================
        
        general_recommendations = [
            f"Always validate your model on new, unseen data before making business decisions.",
            f"Monitor your model's performance regularly as new data arrives.",
            f"Consider ensemble methods if you need higher accuracy (combining multiple models).",
            f"Use feature scaling and normalization to ensure all features contribute equally.",
            f"Regularly retrain your model with updated data to maintain performance."
        ]
        
        if best_model_metrics.get('f1_score', 0) < 0.7 or best_model_metrics.get('r2_score', 0) < 0.7:
            general_recommendations.insert(0, "Consider improving data quality or collecting more features - current model performance could be better.")
        
        # ============================================================================
        # PART 8: DATASET TYPE DETECTION & BUSINESS RECOMMENDATIONS
        # ============================================================================
        
        # Detect dataset type based on column names and target
        dataset_type, confidence = detect_dataset_type(df, target_column, X)
        
        # Generate business-specific recommendations
        business_recommendations = generate_business_recommendations(
            dataset_type=dataset_type,
            target_column=target_column,
            df=df,
            X=X,
            y=y,
            feature_importance=feature_importance_list,
            problem_type=problem_type
        )
        
        business_insights = generate_business_insights(
            dataset_type=dataset_type,
            target_column=target_column,
            df=df,
            problem_type=problem_type
        )
        
        # ============================================================================
        # RETURN RESULTS
        # ============================================================================
        
        return {
            "success": True,
            "problem_type": problem_type,
            "problem_type_analysis": problem_type_analysis,
            "feature_importance": feature_importance_list[:15],  # Top 15 features
            "feature_recommendations": feature_recommendations[:10],  # Top 10 recommendations
            "data_suggestions": data_suggestions,
            "model_selection_reason": {
                "reasons": model_selection_reasons
            },
            "key_insights": key_insights,
            "general_recommendations": general_recommendations,
            "dataset_type": dataset_type,
            "dataset_type_confidence": confidence,
            "business_recommendations": business_recommendations,
            "business_insights": business_insights
        }
        
    except Exception as e:
        print(f"Error in get_ai_suggestions: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Error generating AI suggestions: {str(e)}"}

@app.post("/generate-ai-insights")
async def generate_ai_insights(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI-powered business insights using LLM and SHAP analysis"""
    try:
        global cleaned_dataframe, trained_models, best_model, ai_advisor, latest_ai_insights_report

        if cleaned_dataframe is None or trained_models is None or best_model is None:
            return {"success": False, "error": "No trained model available. Please train a model first."}

        request_api_key = str(request.get('llm_api_key', request.get('openai_api_key', ''))).strip()
        request_provider = str(request.get('llm_provider', '')).strip() or None

        # Initialize AI Advisor if not already done, or refresh if caller provided a key.
        if ai_advisor is None or request_api_key or request_provider:
            try:
                ai_advisor = AIAdvisor(llm_api_key=request_api_key or None, provider=request_provider)
            except ValueError as e:
                return {"success": False, "error": str(e)}

        target_column = request.get('target_column')
        dataset_type = request.get('dataset_type', 'General')
        include_historical_analysis = request.get('include_historical_analysis', False)

        if not target_column:
            return {"success": False, "error": "Target column must be specified"}

        if target_column not in cleaned_dataframe.columns:
            return {"success": False, "error": f"Target column '{target_column}' not found in dataset"}

        df = cleaned_dataframe.copy()
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Get the best model
        model = trained_models.get(best_model)
        if model is None:
            return {"success": False, "error": f"Best model '{best_model}' not found"}

        # Prepare data for analysis
        X_prepared = X.copy()

        # Encode categorical variables (same as training)
        for col in X_prepared.columns:
            if not pd_types.is_numeric_dtype(X_prepared[col]):
                # Use label encoding for consistency
                le = LabelEncoder()
                X_prepared[col] = le.fit_transform(X_prepared[col].astype(str))

        # Generate predictions for analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_prepared)
        predictions = model.predict(X_scaled)

        # Prepare historical data if requested
        historical_data = None
        if include_historical_analysis:
            historical_data = df.copy()  # Use current data as historical for now

        # Generate comprehensive AI insights
        insights_report = ai_advisor.generate_comprehensive_report(
            model=model,
            X_data=X_prepared,
            y_true=y,
            predictions=predictions,
            target_column=target_column,
            dataset_type=dataset_type,
            historical_data=historical_data
        )

        if not insights_report["success"]:
            return insights_report

        # Keep latest report fragments for RAG indexing.
        latest_ai_insights_report = {
            "feature_importance": insights_report.get("feature_importance", []),
            "shap_summary": insights_report.get("shap_summary", {}),
            "insights": insights_report.get("insights", []),
            "predictions_summary": insights_report.get("predictions_summary", {}),
            "risk_assessment": insights_report.get("risk_assessment", {}),
            "trend_analysis": insights_report.get("trend_analysis", {}),
            "target_column": target_column,
            "dataset_type": dataset_type,
            "generated_at": datetime.now().isoformat()
        }

        insight_items = insights_report.get("insights", [])
        if insight_items:
            failed_insights = [
                item for item in insight_items
                if str(item.get("content", "")).lower().startswith("unable to generate")
            ]
            if len(failed_insights) == len(insight_items):
                return {
                    "success": False,
                    "error": failed_insights[0].get(
                        "content",
                        "LLM insight generation failed. Verify API key, model access, and account quota."
                    )
                }

        # Format the response for the frontend
        response = {
            "success": True,
            "insights": insights_report["insights"],
            "feature_importance": insights_report["feature_importance"][:10],  # Top 10 features
            "predictions_summary": insights_report["predictions_summary"],
            "shap_summary": insights_report["shap_summary"],
            "dataset_type": dataset_type,
            "target_column": target_column
        }

        # Add optional components if available
        if "trend_analysis" in insights_report:
            response["trend_analysis"] = insights_report["trend_analysis"]

        if "risk_assessment" in insights_report:
            response["risk_assessment"] = insights_report["risk_assessment"]

        if "implementation_roadmap" in insights_report:
            response["implementation_roadmap"] = insights_report["implementation_roadmap"]

        return response

    except Exception as e:
        logger.error(f"Error in generate_ai_insights: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Error generating AI insights: {str(e)}"}


def _assemble_rag_sources() -> Dict[str, Any]:
    """Collect all available sources for the RAG index."""
    global current_dataframe, cleaned_dataframe, ml_results, best_model, latest_ai_insights_report

    historical_dataset = cleaned_dataframe if cleaned_dataframe is not None else current_dataframe

    ml_outputs_payload = {
        "best_model": best_model,
        "ml_results": ml_results or {},
    }

    shap_payload = latest_ai_insights_report or {
        "note": "SHAP explanations are not available yet. Run /generate-ai-insights after ML training."
    }

    return {
        "historical_dataset": historical_dataset,
        "ml_outputs": ml_outputs_payload,
        "shap_explanations": shap_payload,
    }


def _summarize_dataset_for_chat(df: pd.DataFrame) -> Dict[str, Any]:
    """Build compact dataset summary for chatbot context."""
    if df is None or df.empty:
        return {}

    stats = calculate_dataset_stats(df)
    summary = {
        "num_rows": stats.get("num_rows"),
        "num_columns": stats.get("num_columns"),
        "columns": stats.get("columns", []),
        "missing_percentage": stats.get("missing_percentage"),
        "num_duplicates": stats.get("num_duplicates"),
        "numeric_summary": stats.get("numeric_summary", {}),
        "categorical_summary": stats.get("categorical_summary", {}),
    }
    return _json_safe(summary)


def _build_model_info_for_chat() -> Dict[str, Any]:
    """Build model metadata and performance context for chatbot."""
    global best_model, ml_results, training_artifacts

    results = ml_results or {}
    best_metrics = results.get(best_model, {}) if best_model else {}
    derived_feature_importance = _derive_feature_importance_for_chat()

    return _json_safe(
        {
            "best_model": best_model,
            "best_metrics": best_metrics,
            "all_model_results": results,
            "problem_type": (
                "classification" if (training_artifacts or {}).get("is_classification") else "regression"
                if training_artifacts
                else None
            ),
            "target_column": (training_artifacts or {}).get("target_column"),
            "derived_feature_importance": derived_feature_importance,
        }
    )


def _derive_feature_importance_for_chat() -> list:
    """Derive feature importance from trained best model when SHAP report is unavailable."""
    global trained_models, best_model, training_artifacts

    try:
        if not trained_models or not best_model:
            return []

        model = trained_models.get(best_model)
        if model is None:
            return []

        feature_columns = (training_artifacts or {}).get("feature_columns", [])
        if not feature_columns:
            return []

        raw_importance = None
        if hasattr(model, "feature_importances_"):
            raw_importance = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(getattr(model, "coef_"), dtype=float)
            if coef.ndim == 1:
                raw_importance = np.abs(coef)
            else:
                raw_importance = np.abs(coef).mean(axis=0)

        if raw_importance is None or raw_importance.size == 0:
            return []

        size = min(len(feature_columns), int(raw_importance.shape[0]))
        if size <= 0:
            return []

        items = []
        for i in range(size):
            items.append(
                {
                    "feature": str(feature_columns[i]),
                    "importance": float(raw_importance[i]),
                    "source": "model_derived",
                }
            )

        items.sort(key=lambda x: x["importance"], reverse=True)
        for idx, item in enumerate(items, start=1):
            item["rank"] = idx

        return _json_safe(items)
    except Exception:
        return []


def _build_chatbot_context(retrieved_context: Any = None) -> Dict[str, Any]:
    """Assemble structured context consumed by generate_chatbot_response."""
    global current_dataframe, cleaned_dataframe, latest_ai_insights_report

    source_df = cleaned_dataframe if cleaned_dataframe is not None else current_dataframe
    insights = latest_ai_insights_report or {}
    fallback_feature_importance = _derive_feature_importance_for_chat()
    feature_importance = insights.get("feature_importance", []) or fallback_feature_importance

    context = {
        "dataset_summary": _summarize_dataset_for_chat(source_df) if source_df is not None else {},
        "model_info": _build_model_info_for_chat(),
        "shap_summary": {
            "shap_summary": insights.get("shap_summary", {}),
            "feature_importance": feature_importance,
            "insights": insights.get("insights", []),
            "feature_importance_source": (
                "shap" if insights.get("feature_importance") else "model_derived" if fallback_feature_importance else "none"
            ),
        },
        "predictions_summary": insights.get("predictions_summary", {}),
        "trend_analysis": insights.get("trend_analysis", {}),
        "retrieved_context": retrieved_context or [],
    }
    return _json_safe(context)


def _build_what_if_input(
    request: Dict[str, Any],
    feature_columns: list,
    feature_defaults: Dict[str, Any],
    source_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Create baseline input row from request data, row index, or defaults."""
    raw_input = request.get("input_data") or {}
    if isinstance(raw_input, dict) and raw_input:
        base = {col: feature_defaults.get(col) for col in feature_columns}
        for col in feature_columns:
            if col in raw_input:
                base[col] = raw_input.get(col)
        return base

    row_index = request.get("row_index")
    if row_index is not None:
        try:
            idx = int(row_index)
            if 0 <= idx < len(source_df):
                row_data = source_df.iloc[idx].to_dict()
                return {col: row_data.get(col, feature_defaults.get(col)) for col in feature_columns}
        except Exception:
            pass

    # By default use first dataset row so scenarios are based on real data.
    if source_df is not None and len(source_df) > 0:
        row_data = source_df.iloc[0].to_dict()
        return {col: row_data.get(col, feature_defaults.get(col)) for col in feature_columns}

    return {col: feature_defaults.get(col) for col in feature_columns}


@app.post("/chatbot/build-index")
async def build_chatbot_index(request: Dict[str, Any]) -> Dict[str, Any]:
    """Build (or rebuild) FAISS index for the analytics chatbot."""
    try:
        global rag_chatbot

        request_api_key = str(request.get('llm_api_key', '')).strip() or None
        request_provider = str(request.get('llm_provider', '')).strip() or None
        embedding_model = str(request.get('embedding_model', 'all-MiniLM-L6-v2')).strip()

        rag_chatbot = BusinessAnalyticsRAG(
            embedding_model=embedding_model,
            llm_provider=request_provider,
            llm_api_key=request_api_key,
        )

        rag_data = _assemble_rag_sources()
        result = rag_chatbot.create_vector_store(rag_data)

        return {
            "success": True,
            "message": "Vector index created successfully",
            "index_details": result,
        }
    except Exception as e:
        logger.error(f"Error building chatbot index: {str(e)}")
        return {"success": False, "error": f"Error building chatbot index: {str(e)}"}


@app.post("/chatbot/query")
async def chatbot_query(request: Dict[str, Any]) -> Dict[str, Any]:
    """Answer natural language questions via retrieval-augmented generation."""
    try:
        global rag_chatbot

        query = str(request.get('query', '')).strip()
        if not query:
            return {"success": False, "error": "Query must be provided."}

        request_api_key = str(request.get('llm_api_key', '')).strip() or None
        request_provider = str(request.get('llm_provider', '')).strip() or None

        # Recreate client if caller provides provider/key overrides.
        if rag_chatbot is None or request_api_key or request_provider:
            rag_chatbot = BusinessAnalyticsRAG(
                embedding_model='all-MiniLM-L6-v2',
                llm_provider=request_provider,
                llm_api_key=request_api_key,
            )

        # Auto-create index if missing.
        if rag_chatbot.index is None or not rag_chatbot.chunks:
            rag_data = _assemble_rag_sources()
            rag_chatbot.create_vector_store(rag_data)

        retrieval_payload = rag_chatbot.retrieve_context(query)
        chat_context = _build_chatbot_context(retrieval_payload.get("contexts", []))
        answer_payload = rag_chatbot.generate_chatbot_response(query, chat_context)

        return {
            "success": bool(answer_payload.get("success", False)),
            "query": query,
            "answer": answer_payload.get("answer", ""),
            "model": answer_payload.get("model"),
            "retrieved_context": retrieval_payload.get("contexts", []),
            "response_mode": "context_aware",
        }
    except Exception as e:
        logger.error(f"Error in chatbot query: {str(e)}")
        return {"success": False, "error": f"Error processing chatbot query: {str(e)}"}


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


@app.post("/what-if-config")
async def get_what_if_config(request: Dict[str, Any]) -> Dict[str, Any]:
    """Provide what-if metadata: feature types, categorical values, and selected row preview."""
    try:
        global cleaned_dataframe, current_dataframe, training_artifacts

        artifacts = training_artifacts or {}
        feature_columns = artifacts.get("feature_columns", [])
        if not feature_columns:
            return {
                "success": False,
                "error": "Training artifacts are missing. Train a model first."
            }

        feature_types = artifacts.get("feature_types", {})
        category_options = artifacts.get("category_options", {})

        base_df = cleaned_dataframe if cleaned_dataframe is not None else current_dataframe
        if base_df is None:
            return {"success": False, "error": "Dataset not available."}

        source_features_df = base_df.drop(columns=[artifacts.get("target_column")], errors='ignore')

        selected_row = 0
        requested_row = request.get("row_index")
        if requested_row is not None:
            try:
                selected_row = int(requested_row)
            except Exception:
                selected_row = 0

        if len(source_features_df) > 0:
            selected_row = max(0, min(selected_row, len(source_features_df) - 1))
            row_preview = source_features_df.iloc[selected_row].to_dict()
        else:
            row_preview = {}

        return {
            "success": True,
            "row_count": int(len(source_features_df)),
            "default_row_index": selected_row,
            "feature_columns": feature_columns,
            "feature_types": _json_safe(feature_types),
            "category_options": _json_safe(category_options),
            "row_preview": _json_safe(row_preview),
        }
    except Exception as e:
        logger.error(f"Error in get_what_if_config: {str(e)}")
        return {"success": False, "error": f"Error loading what-if config: {str(e)}"}


@app.post("/simulate-what-if")
async def simulate_what_if(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run what-if simulation by applying feature changes and comparing predictions."""
    try:
        global cleaned_dataframe, current_dataframe, trained_models, best_model, training_artifacts

        if trained_models is None or best_model is None:
            return {"success": False, "error": "No trained model available. Please train a model first."}

        model = trained_models.get(best_model)
        if model is None:
            return {"success": False, "error": f"Best model '{best_model}' not found."}

        artifacts = training_artifacts or {}
        feature_columns = artifacts.get("feature_columns", [])
        label_encoders = artifacts.get("label_encoders", {})
        scaler = artifacts.get("scaler")
        is_classification = bool(artifacts.get("is_classification", False))
        target_column = artifacts.get("target_column", "target")
        feature_defaults = artifacts.get("feature_defaults", {})

        if not feature_columns:
            return {
                "success": False,
                "error": "Training artifacts are missing feature metadata. Retrain the model and try again."
            }

        changes = request.get("changes") or {}
        if not isinstance(changes, dict) or not changes:
            return {"success": False, "error": "changes must be a non-empty object."}

        base_df = cleaned_dataframe if cleaned_dataframe is not None else current_dataframe
        if base_df is None:
            return {"success": False, "error": "Dataset not available. Upload data before simulation."}

        source_features_df = base_df.drop(columns=[target_column], errors='ignore')

        baseline_input = _build_what_if_input(
            request=request,
            feature_columns=feature_columns,
            feature_defaults=feature_defaults,
            source_df=source_features_df,
        )

        simulation = simulate_scenario(
            model=model,
            input_data=baseline_input,
            changes=changes,
            feature_columns=feature_columns,
            label_encoders=label_encoders,
            scaler=scaler,
            is_classification=is_classification,
        )

        validation = simulation.get("validation", {})
        input_validation = validation.get("new_input", {})
        if input_validation:
            return {
                "success": False,
                "error": "One or more feature values are invalid for this dataset.",
                "validation": input_validation,
            }

        old_pred = simulation.get("old_prediction", {})
        new_pred = simulation.get("new_prediction", {})
        comparison = compare_results(old_pred, new_pred)

        request_api_key = str(request.get('llm_api_key', '')).strip() or None
        request_provider = str(request.get('llm_provider', '')).strip() or None

        explanation = generate_explanation(
            changes=simulation.get("applied_changes", {}),
            old_pred=old_pred,
            new_pred=new_pred,
            comparison=comparison,
            target_label=target_column,
            llm_provider=request_provider,
            llm_api_key=request_api_key,
        )

        return {
            "success": True,
            "model": best_model,
            "target_column": target_column,
            "input_before": simulation.get("input_before", {}),
            "input_after": simulation.get("input_after", {}),
            "applied_changes": simulation.get("applied_changes", {}),
            "old_prediction": old_pred,
            "new_prediction": new_pred,
            "comparison": comparison,
            "insight": explanation,
        }
    except Exception as e:
        logger.error(f"Error in simulate_what_if: {str(e)}")
        return {"success": False, "error": f"Error running what-if simulation: {str(e)}"}

@app.get("/project-state")
async def get_project_state() -> Dict[str, Any]:
    """Get the current state of the entire project for chatbot contextual help."""
    try:
        global current_dataframe, cleaned_dataframe, trained_models, best_model, ml_results, training_artifacts
        
        state = {
            "workflow_stage": "initial",
            "available_features": [],
            "next_suggested_step": "",
            "dataset_info": None,
            "cleaning_info": None,
            "training_info": None,
            "suggestions_available": False,
            "chatbot_ready": False
        }
        
        # Stage 1: Dataset Upload
        if current_dataframe is not None:
            state["workflow_stage"] = "data_loaded"
            stats = calculate_dataset_stats(current_dataframe)
            state["dataset_info"] = {
                "rows": stats.get("num_rows"),
                "columns": stats.get("num_columns"),
                "column_names": stats.get("columns", []),
                "missing_percentage": stats.get("missing_percentage"),
                "duplicates": stats.get("num_duplicates"),
            }
            state["next_suggested_step"] = "Explore your dataset → Click 'Clean Data' to start data cleaning"
            state["available_features"].append("dataset_exploration")
        
        # Stage 2: Data Cleaning
        if cleaned_dataframe is not None:
            state["workflow_stage"] = "data_cleaned"
            clean_stats = calculate_dataset_stats(cleaned_dataframe)
            state["cleaning_info"] = {
                "rows": clean_stats.get("num_rows"),
                "columns": clean_stats.get("num_columns"),
            }
            state["next_suggested_step"] = "Train machine learning models on your cleaned data"
            state["available_features"].extend(["data_cleaning", "download_cleaned"])
        
        # Stage 3: ML Training
        if trained_models is not None and best_model is not None:
            state["workflow_stage"] = "model_trained"
            state["training_info"] = {
                "best_model": best_model,
                "num_models_trained": len(trained_models),
                "models_tested": list(trained_models.keys()),
            }
            if ml_results:
                best_metrics = ml_results.get(best_model, {})
                state["training_info"]["best_model_performance"] = best_metrics
            state["available_features"].extend(["ml_training", "model_comparison"])
            state["next_suggested_step"] = "Generate AI insights and get business recommendations"
            state["suggestions_available"] = True
        
        # Stage 4: AI Insights & Chatbot Ready
        if latest_ai_insights_report and len(latest_ai_insights_report) > 0:
            state["workflow_stage"] = "ai_insights_generated"
            state["available_features"].extend(["ai_insights", "what_if_simulation", "rag_chatbot"])
            state["chatbot_ready"] = True
            state["next_suggested_step"] = "Ask me questions about your data using the chatbot! Try: 'What are the key factors in my dataset?' or 'How can I improve predictions?'"
        
        return {
            "success": True,
            "project_state": state
        }
    
    except Exception as e:
        logger.error(f"Error in get_project_state: {str(e)}")
        return {"success": False, "error": str(e)}


@app.post("/chatbot/answer-question")
async def answer_question(request: Dict[str, Any]) -> Dict[str, Any]:
    """Answer project questions with context-aware reasoning from real artifacts."""
    try:
        global current_dataframe, cleaned_dataframe, trained_models, best_model, rag_chatbot

        question = str(request.get('question', '')).strip()
        if not question:
            return {"success": False, "error": "Question must be provided"}

        # Determine project stage
        stage = "initial"
        if latest_ai_insights_report and len(latest_ai_insights_report) > 0:
            stage = "ai_insights_generated"
        elif trained_models is not None and best_model is not None:
            stage = "model_trained"
        elif cleaned_dataframe is not None:
            stage = "data_cleaned"
        elif current_dataframe is not None:
            stage = "data_loaded"

        request_api_key = str(request.get('llm_api_key', '')).strip() or None
        request_provider = str(request.get('llm_provider', '')).strip() or None

        if rag_chatbot is None or request_api_key or request_provider:
            rag_chatbot = BusinessAnalyticsRAG(
                embedding_model='all-MiniLM-L6-v2',
                llm_provider=request_provider,
                llm_api_key=request_api_key,
            )

        retrieval_context = []
        try:
            if rag_chatbot.index is None or not rag_chatbot.chunks:
                rag_data = _assemble_rag_sources()
                rag_chatbot.create_vector_store(rag_data)
            retrieval_payload = rag_chatbot.retrieve_context(question)
            retrieval_context = retrieval_payload.get("contexts", [])
        except Exception:
            retrieval_context = []

        chat_context = _build_chatbot_context(retrieval_context)
        answer_payload = rag_chatbot.generate_chatbot_response(question, chat_context)
        answer_text = str(answer_payload.get("answer", "")).strip() or "Not enough data available"

        fallback_to_rag = False
        helper_message = None
        if answer_text == "Not enough data available" and stage != "ai_insights_generated":
            fallback_to_rag = False
            helper_message = (
                "I need more project context to answer this precisely. "
                "Complete ML training and generate AI insights, then ask again."
            )

        return {
            "success": True,
            "answer": answer_text,
            "stage": stage,
            "fallback_to_rag": fallback_to_rag,
            "message": helper_message,
            "response_mode": "context_aware"
        }
    
    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        return {"success": False, "error": str(e), "fallback_to_rag": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
