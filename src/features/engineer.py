import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# -----------------------------
# Feature Engineering
# -----------------------------
def classify_age_group(age):
    if 18 <= age <= 24:
        return "Young Adult"
    elif 24 < age <= 39:
        return "Adult"
    elif 39 < age <= 59:
        return "Mid-Career"
    else:
        return "Senior"


def classify_interaction_frequency(x):
    if 0 < x <= 7:
        return "Highly Active"
    elif 7 < x <= 15:
        return "Active"
    else:
        return "Dormant"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Age Group"] = df["Age"].apply(classify_age_group)
    df["Interaction Frequency"] = df["Last Interaction"].apply(
        classify_interaction_frequency
    )

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Age Group"] = X["Age"].apply(classify_age_group)
        X["Interaction Frequency"] = X["Last Interaction"].apply(
            classify_interaction_frequency
        )

        return X


# -----------------------------
# Preprocessor
# -----------------------------
def create_preprocessor():
    numerical_features = [
        'Age', 'Support Calls', 'Payment Delay',
        'Last Interaction', 'Total Spend'
    ]

    categorical_features = [
        'Gender', 'Age Group', 'Interaction Frequency'
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ])

    return preprocessor