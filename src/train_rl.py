# src/train_rl.py
"""
Train an offline RL agent using d3rlpy (CQL) if available.
If d3rlpy is not installed, fallback: evaluate a simple threshold policy
based on the supervised model's predicted probability.
Usage: python src/train_rl.py
"""
import os
import joblib
import numpy as np
from preprocess import load_df, preprocess
from rl_env import build_rl_dataset_from_df

ARTIFACTS = "../artifacts"
DATA_PATH = "../data/accepted_2007_to_2018.csv"

def try_train_d3rlpy(dataset):
    try:
        import d3rlpy
        from d3rlpy.algos import CQL
        from d3rlpy.dataset import MDPDataset
        obs = dataset["observations"]
        acts = dataset["actions"].astype(float)
        rews = dataset["rewards"].squeeze()
        terms = dataset["terminals"].squeeze().astype(bool)
        # d3rlpy expects lists of episodes; we will treat each row as single-step episode
        mdp = MDPDataset(observations=obs, actions=acts, rewards=rews, terminals=terms)
        algo = CQL(actor_layers=[256,256], q_func_layers=[256,256], use_gpu=False)
        algo.fit(mdp, n_epochs=20, scorers={"average_value": d3rlpy.metrics.average_value_scorer})
        # Save policy
        algo.save_model(os.path.join(ARTIFACTS, "cql_model"))
        print("CQL training done and saved.")
        return True
    except Exception as e:
        print("d3rlpy training failed or not installed:", e)
        return False

def fallback_threshold_policy(dataset):
    # Simple fallback: load supervised model and approve if predicted default prob < 0.5
    try:
        import torch
        from dl_model import MLP
        model_path = os.path.join(ARTIFACTS, "best_mlp.pt")
        feature_names = joblib.load(os.path.join(ARTIFACTS, "feature_names.joblib"))
        # load X and y used earlier (we re-preprocess a small sample)
        df = load_df(DATA_PATH)
        X, y, _ = preprocess(df, sample_frac=0.05, save_dir=ARTIFACTS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=X.shape[1])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X).float()).numpy()
            probs = 1 / (1 + np.exp(-logits))
        # policy: approve when prob(default) < 0.5
        actions = (probs < 0.5).astype(int)
        # estimate policy value on dataset
        loan_amnt = df["loan_amnt"].astype(float).fillna(0.0).values[:len(actions)]
        int_rate = df["int_rate"].astype(str).str.rstrip("%").astype(float).fillna(0.0).values[:len(actions)] / 100.0
        rewards = []
        for i, a in enumerate(actions):
            if a == 0:
                rewards.append(0.0)
            else:
                if y[i] == 0:
                    rewards.append(loan_amnt[i] * int_rate[i])
                else:
                    rewards.append(-loan_amnt[i])
        epv = np.mean(rewards)
        print(f"Fallback policy EPV (mean per-applicant reward): {epv:.4f}")
        return True
    except Exception as e:
        print("Fallback threshold policy failed:", e)
        return False

def train_rl():
    os.makedirs(ARTIFACTS, exist_ok=True)
    print("Preparing RL dataset...")
    df = load_df(DATA_PATH)
    X, y, feat = preprocess(df, sample_frac=0.05, save_dir=ARTIFACTS)
    dataset = build_rl_dataset_from_df(df.sample(frac=0.05, random_state=42).reset_index(drop=True), X, y, feat)
    ok = try_train_d3rlpy(dataset)
    if not ok:
        print("Trying fallback threshold policy evaluate...")
        fallback_threshold_policy(dataset)

if __name__ == "__main__":
    train_rl()
