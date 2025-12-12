#Shodh Project - Loan Default Prediction & Offline Reinforcement Learning for Profit Optimization

This project implements two complementary models for financial decision-making using the LendingClub dataset:

1. A Predictive Deep Learning Model (Supervised Learning)
2. An Offline Reinforcement Learning Agent (CQL / Fallback Policy Evaluation)

The goal is to compare risk-based decision-making vs reward-based decision-making, and understand which approach maximizes
business profit.

#Project Structure

Shodh-Project/
#data/ ← Raw dataset (ignored in GitHub)

#notebooks/
   ├── 1_eda.ipynb    ← Exploratory Data Analysis notebook
   ├── 2_supervised_model.ipynb ← Deep learning training notebook
   └── 3_offline_rl.ipynb   ← Offline RL notebook

#src/
   ├── preprocess.py            ← Cleaning, feature engineering, encoding
   ├── dl_model.py              ← MLP classifier model
   ├── train_supervised.py      ← Train + evaluate DL model
   ├── rl_env.py                ← Defines RL states, actions, rewards
   ├── train_rl.py              ← RL dataset preparation + fallback policy EPV
   └── train_rl_d3rlpy.py       ← (Optional) Train offline RL using d3rlpy

# artifacts/    ← Saved models & preprocessors
│── final_report_full_with_analysis.pdf
│── requirements.txt
│── README.md


1. Exploratory Data Analysis (EDA)

A large sample (200k rows) of the LendingClub dataset was analyzed.

#Key insights:

Defaults are rare → highly imbalanced problem
Interest rate strongly predicts default
Higher credit grades (A/B) show low default risk
Lower grades (E/F/G) show high default risk
Loan amount is right-skewed
DTI, FICO, and employment length correlate with repayment performance**

EDA confirms this is a multi-factor risk prediction problem with significant imbalance.

2. Predictive Deep Learning Model

A Multi-Layer Perceptron (MLP) is trained on engineered features:

#Model Performance:

| Metric       | Score      |
| ------------ | ---------- |
|   AUC        |  0.7065    |
|   F1-Score   |   0.3925   |
|   Accuracy   |   0.5975   |

# Interpretation:

AUC indicates moderate ranking ability
F1-score shows improved detection of default cases after class weighting
Accuracy is not meaningful due to imbalance
DL model is good for risk classification, but not for profit optimization

3. Offline Reinforcement Learning Agent

The RL agent is trained using **static historical data**:

#State (s):

 Vector of preprocessed loan applicant features

#Action (a):

* 0 = Deny Loan
* 1 = Approve Loan

### **Reward (r):**

* Deny → 0
* Approve & Paid → + loan_amnt * int_rate
* Approve & Default → − loan_amnt

This aligns the model with **financial profit**, not prediction accuracy.

### ⭐ RL Evaluation (Fallback Policy EPV):

| Metric                           | Value         |
| -------------------------------- | ------------- |
| Estimated Policy Value (EPV)     |  +275.6698    |

A positive EPV means the policy **generates profit per applicant**.


4. DL Policy vs RL Policy — Key Differences

#Deep Learning Model

* Approves loans with predicted default probability < threshold
* Focuses on estimating risk
* Conservative, avoids moderate-risk profiles
* Not aware of potential profit

#Reinforcement Learning Policy

* Approves loans with **positive expected reward**, even if risky
* Focuses on *maximizing financial value*
* Will approve high-interest loans with moderate risk
* Balances profit vs loss mathematically

 Example:

| Applicant Feature                | DL Decision | RL Decision | Reason                                    |
| -------------------------------- | ----------- | ----------- | ----------------------------------------- |
| High interest (20%), medium risk |  Deny      |  Approve   | RL sees high expected reward despite risk |


5. Conclusion & Recommendations

 RL-based policies outperform DL classifiers **for business profit**
 DL is still valuable for **risk ranking and interpretability**
 Combining both approaches can yield a **hybrid decision engine**


6. Future Work

Modeling Improvements

* Train real **CQL/IQL** models using d3rlpy (Linux/WSL recommended)
* Try **XGBoost, LightGBM, CatBoost** for risk prediction
* Add **threshold tuning for profit**, not accuracy

Data Improvements

* Monthly repayment behavior
* Recovery amounts on defaults
* Employment stability history
* External credit bureau scores

System Improvements
* Fairness and bias evaluation
* Profit-calibrated models
* A/B testing for policy deployment

 
7. How to Run the Project

#Install dependencies:
pip install -r requirements.txt

# Train supervised model:
python src/train_supervised.py


#Train RL agent (fallback policy evaluation):
python src/train_rl.py


#Train RL agent:
python src/train_rl.py

8. Final Report

#Author
Lakshya Sharma
Machine Learning & Reinforcement Learning Project — Shodh Assignment

