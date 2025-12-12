# src/preprocess.py
"""
Load and preprocess LendingClub accepted_2007_to_2018.csv
Produces: X (numpy), y (numpy), feature_names (list)
Saves fitted scalers/encoders to disk (joblib)
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "term",
    "grade", "sub_grade", "purpose", "emp_length",
    "home_ownership"
]

def load_df(path, sample_size=200000):
    """
    Loads only a sample (e.g., 200k rows) from a large CSV to avoid memory issues.
    """
    chunksize = 50000
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= sample_size:
            break
    df = pd.concat(chunks, ignore_index=True)
    return df

def build_target(df):
    # binary target: 1 = defaulted/charged off, 0 = fully paid
    def is_default(s):
        if pd.isna(s):
            return 0
        s = str(s).lower()
        if "charged off" in s or "default" in s or "late" in s:
            return 1
        return 0
    return df["loan_status"].apply(is_default).astype(int).values

def preprocess(df, sample_frac=None, save_dir=None):
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # select columns (drop if missing)
    cols = [c for c in FEATURES if c in df.columns]
    df = df[cols + ["loan_status", "loan_amnt"]]  # keep loan_status and amount
    # target
    y = build_target(df)

    # Simple cleaning / create numeric features
    # Clean interest rate column if it's like '13.56%'
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.rstrip("%").astype(float)

    # emp_length: normalize text like '10+ years' or '< 1 year'
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].astype(str).replace("n/a", "")
        df["emp_length"] = df["emp_length"].str.extract(r'(\d+)').astype(float).fillna(0.0)

    # term: '36 months' -> 36
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r'(\d+)').astype(float)

    # numeric and categorical split
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # remove loan_status
    if "loan_status" in numeric_cols:
        numeric_cols.remove("loan_status")

    categorical_cols = [c for c in cols if c not in numeric_cols]

    # Impute numeric
    num_imp = SimpleImputer(strategy="median")
    X_num = num_imp.fit_transform(df[numeric_cols])

    # Encode categoricals
    if len(categorical_cols) > 0:
        cat_imp = SimpleImputer(strategy="constant", fill_value="missing")
        X_cat_raw = cat_imp.fit_transform(df[categorical_cols].astype(str))
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = ohe.fit_transform(X_cat_raw)
    else:
        X_cat = np.zeros((len(df), 0))
        ohe = None
        cat_imp = None

    # Scale numeric
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Concatenate
    X = np.hstack([X_num_scaled, X_cat])
    feature_names = list(numeric_cols)
    if ohe:
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names += ohe_names

    # Save preprocessors
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(num_imp, os.path.join(save_dir, "num_imputer.joblib"))
        joblib.dump(cat_imp, os.path.join(save_dir, "cat_imputer.joblib"))
        joblib.dump(ohe, os.path.join(save_dir, "ohe.joblib"))
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
        joblib.dump(feature_names, os.path.join(save_dir, "feature_names.joblib"))

    return X, y.astype(int), feature_names
