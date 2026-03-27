"""
Test script for the AI-Powered Reasoning and Advisory Engine

This script demonstrates how to use the AIAdvisor class to generate
business insights from ML model predictions and SHAP explanations.

Usage:
1. Set your OPENAI_API_KEY environment variable
2. Run: python test_ai_advisor.py

Requirements:
- Trained ML model (saved as pickle or similar)
- Test dataset (pandas DataFrame)
- OpenAI API key
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Import the AI Advisor
from ai_advisor import AIAdvisor

def create_sample_data():
    """Create a sample customer churn dataset for demonstration"""
    np.random.seed(42)

    # Generate sample data
    n_samples = 1000

    data = {
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'complaints_count': np.random.poisson(0.5, n_samples),
        'support_calls': np.random.poisson(1, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    }

    df = pd.DataFrame(data)

    # Create target variable (churn) based on features
    churn_probability = (
        (df['tenure_months'] < 12) * 0.3 +
        (df['complaints_count'] > 2) * 0.4 +
        (df['contract_type'] == 'Month-to-month') * 0.2 +
        (df['payment_method'] == 'Electronic check') * 0.1 +
        np.random.uniform(0, 0.2, n_samples)  # Random noise
    )

    df['churn'] = (churn_probability > 0.5).astype(int)

    return df

def train_sample_model(df):
    """Train a simple Random Forest model on the sample data"""
    # Prepare features
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, encoders

def main():
    """Main demonstration function"""
    print("🤖 AI-Powered Reasoning and Advisory Engine Demo")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("Windows: $env:OPENAI_API_KEY = 'your-key-here'")
        print("Linux/macOS: export OPENAI_API_KEY=your-key-here")
        return

    try:
        # Create sample data
        print("📊 Creating sample customer churn dataset...")
        df = create_sample_data()
        print(f"✅ Created dataset with {len(df)} samples and {len(df.columns)} features")

        # Train model
        print("🤖 Training Random Forest model...")
        model, X_test, y_test, encoders = train_sample_model(df)
        print("✅ Model trained successfully")

        # Initialize AI Advisor
        print("🧠 Initializing AI Advisor...")
        advisor = AIAdvisor()
        print("✅ AI Advisor initialized")

        # Generate predictions
        print("🔮 Generating predictions...")
        predictions = model.predict(X_test.values)
        print(f"✅ Generated {len(predictions)} predictions")

        # Analyze with AI Advisor
        print("🧠 Generating AI-powered business insights...")
        print("(This may take 30-60 seconds due to OpenAI API calls)")
        print()

        analysis = advisor.analyze_predictions(
            model=model,
            X_data=X_test,
            y_true=y_test,
            predictions=predictions,
            target_column='churn',
            dataset_type='Customer Analytics'
        )

        if analysis['success']:
            print("✅ AI Analysis completed successfully!")
            print()

            # Display insights
            print("💡 AI-GENERATED BUSINESS INSIGHTS:")
            print("-" * 40)

            for insight in analysis['insights']:
                print(f"\n📋 {insight['type'].replace('_', ' ').title()}:")
                print(f"   {insight['content']}")
                print()

            # Display feature importance
            print("🎯 TOP PREDICTIVE FEATURES:")
            print("-" * 30)
            for i, feature in enumerate(analysis['feature_importance'][:5], 1):
                print(".1f")

            print()
            print("📊 PREDICTIONS SUMMARY:")
            print("-" * 25)
            pred_summary = analysis['predictions_summary']
            print(f"Total Predictions: {pred_summary['num_predictions']}")
            print(".3f")
            print(".3f")

            if 'accuracy' in pred_summary:
                print(".3f")

            print()
            print("🚀 Demo completed! The AI Advisor is working correctly.")
            print("You can now integrate this into your FastAPI application.")

        else:
            print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()