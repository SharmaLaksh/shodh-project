# src/rl_env.py
"""
Helpers to convert the preprocessed dataset into an offline RL dataset:
observations, actions, rewards, terminals.
We assume the historical data are *approved* loans (accepted_...).
Action space: 0 = deny, 1 = approve.
If action==deny -> reward 0.
If approve & loan fully paid -> reward = loan_amnt * int_rate (approx profit).
If approve & default -> reward = -loan_amnt (loss).
"""
import numpy as np
import pandas as pd

def build_rl_dataset_from_df(df_pre_raw, X_features, y, feature_names):
    """
    df_pre_raw: original dataframe (not fully processed) containing loan_amnt and int_rate
    X_features: numpy array of processed features (same order as feature_names)
    y: binary default array (1 defaulted)
    Returns dict with observations, actions, rewards, terminals
    """
    n = X_features.shape[0]
    # Build actions: historical accepted loans -> treated as approve (1)
    actions = np.ones((n, 1), dtype=int)

    # Extract loan_amnt and int_rate from raw dataframe if present
    loan_amnt = np.zeros(n, dtype=float)
    int_rate = np.zeros(n, dtype=float)
    if "loan_amnt" in df_pre_raw.columns:
        loan_amnt = df_pre_raw["loan_amnt"].astype(float).fillna(0.0).values
    if "int_rate" in df_pre_raw.columns:
        int_rate = df_pre_raw["int_rate"].astype(str).str.rstrip("%").astype(float).fillna(0.0).values / 100.0

    rewards = np.zeros(n, dtype=float)
    for i in range(n):
        if actions[i, 0] == 0:
            rewards[i] = 0.0
        else:
            if y[i] == 0:
                rewards[i] = loan_amnt[i] * int_rate[i]
            else:
                rewards[i] = -loan_amnt[i]

    # terminals: one-step episodes (True at end)
    terminals = np.ones(n, dtype=bool)

    # observations: X_features
    obs = X_features.astype(float)

    dataset = {
        "observations": obs,
        "actions": actions,
        "rewards": rewards.reshape(-1, 1),
        "terminals": terminals.reshape(-1, 1)
    }
    return dataset
