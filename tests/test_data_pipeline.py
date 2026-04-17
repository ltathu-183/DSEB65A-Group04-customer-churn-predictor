import pytest
import pandas as pd
import numpy as np
from src.preprocess.preprocessor import preprocess
from src.features.engineer import FeatureEngineer, create_preprocessor

# -----------------------------
# FIXTURES
# -----------------------------
@pytest.fixture
def dirty_data():
    """Create a 'dirty' DataFrame with logical errors and messy strings."""
    data = {
        "CustomerID": [1, 2, 3, 4, 5, 6],
        "Age": [25, 15, 120, 45, 30, 30], # 15 and 120 are logical errors
        "Gender": ["Male", "female ", "Male", "FEMALE", "Male", "Male"],
        "Tenure": [10, 5, 5, 5, 5, 5],
        "Usage Frequency": [10, 5, 5, 5, 5, 5],
        "Support Calls": [1, 2, 0, 3, 1, 1],
        "Payment Delay": [0, 5, 0, 10, 2, 2],
        "Subscription Type": ["Basic", "Standard", "Premium", "Basic", "Basic", "Basic"],
        "Contract Length": ["Monthly", "Annual", "Monthly", "Monthly", "Monthly", "Monthly"],
        "Total Spend": [100.0, 50.0, 500.0, 200.0, 150.0, 150.0],
        "Last Interaction": [10, 5, 400, 20, 15, 15], # 400 is error (>365)
        "Churn": [0, 1, 0, 1, 0, 0]
    }
    return pd.DataFrame(data)

# -----------------------------
# PREPROCESSING TESTS
# -----------------------------

def test_preprocess_logic_cleaning(dirty_data):
    """Verify that logical errors (age, interaction) are removed."""
    df_clean = preprocess(dirty_data)
    
    # Rows with Age 15, 120 and Last Interaction 400 should be removed
    # Remaining should be index 0, 3, 4 (Index 5 is a duplicate of 4)
    assert df_clean.shape[0] == 3
    assert df_clean["Age"].min() >= 18
    assert df_clean["Age"].max() <= 100
    assert df_clean["Last Interaction"].max() <= 365


def test_preprocess_standardization(dirty_data):
    """Verify that categorical strings are title-cased and stripped."""
    df_clean = preprocess(dirty_data)
    
    # 'female ' should become 'Female'
    # 'FEMALE' should become 'Female'
    assert "Female" in df_clean["Gender"].unique()
    assert "female " not in df_clean["Gender"].unique()


def test_preprocess_column_dropping(dirty_data):
    """Verify that unnecessary columns are dropped."""
    df_clean = preprocess(dirty_data)
    
    dropped_cols = ["CustomerID", "Tenure", "Usage Frequency", "Subscription Type", "Contract Length"]
    for col in dropped_cols:
        assert col not in df_clean.columns


def test_preprocess_type_casting(dirty_data):
    """Verify that columns have the correct data types after processing."""
    df_clean = preprocess(dirty_data)
    
    assert df_clean["Churn"].dtype == int
    assert df_clean["Total Spend"].dtype == float
    assert df_clean["Gender"].dtype == "category"

# -----------------------------
# FEATURE ENGINEERING TESTS
# -----------------------------

def test_feature_engineer_binning():
    """Test if Age Group and Interaction Frequency are binned correctly."""
    fe = FeatureEngineer()
    test_df = pd.DataFrame({
        "Age": [20, 30, 50, 70],
        "Last Interaction": [5, 10, 30, 5]
    })
    
    X_transformed = fe.transform(test_df)
    
    # Age binning check
    assert X_transformed.loc[0, "Age Group"] == "Young Adult"
    assert X_transformed.loc[2, "Age Group"] == "Mid-Career"
    
    # Interaction frequency check
    assert X_transformed.loc[0, "Interaction Frequency"] == "Highly Active"
    assert X_transformed.loc[2, "Interaction Frequency"] == "Dormant"


def test_preprocessor_output_shape():
    """Test if the Scikit-learn preprocessor produces the expected output shape."""
    prep = create_preprocessor()
    
    # Create sample data that looks like the output of FeatureEngineer
    sample_df = pd.DataFrame({
        "Age": [30, 40],
        "Support Calls": [1, 2],
        "Payment Delay": [0, 5],
        "Last Interaction": [10, 20],
        "Total Spend": [100.0, 200.0],
        "Gender": ["Male", "Female"],
        "Age Group": ["Adult", "Mid-Career"],
        "Interaction Frequency": ["Active", "Dormant"]
    })
    
    # Fit and transform
    X_out = prep.fit_transform(sample_df)
    
    # 5 numerical features + encoded categorical features
    # Gender (1 col after drop='first'), Age Group, Interaction freq
    assert X_out.shape[0] == 2
    assert X_out.shape[1] > 5