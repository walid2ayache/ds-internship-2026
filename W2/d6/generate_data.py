"""
Day 6: Telecom Customer Churn Dataset Generator (same data as Day 5)
====================================================================
Generates a realistic, multifaceted Telecom Customer Churn dataset designed for:
1. SciPy Statistical Tests (t-test, Mann-Whitney U, Chi-Square, Pearson/Spearman correlation).
2. The 7-step EDA Checklist:
   shape -> types -> missing -> distributions -> target vs features -> correlations -> leakage check.

Includes intentional, realistic statistical characteristics:
- Numerical features with differing distributions (normal, skewed, heavy-tailed).
- Statistically significant differences between churners and non-churners.
- Subtle data leakage variables to test the 'leakage check' step of the EDA workflow.
- Missing values in specific columns to test missingness diagnostics.

Usage (from the d6 folder):
    python generate_data.py

It writes ``data/telecom_churn.csv`` (3,500 rows x 23 columns, seed 42). The data
folder is gitignored on purpose: anyone can rebuild the exact same file instead
of downloading a binary from the repository.
"""

import numpy as np
import pandas as pd
import os

def generate_telecom_churn_data(n_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Customer Identifiers
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    
    # 2. Demographics
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes",
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45]),
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.10, 0.90]))
    
    # 3. Account & Tenure (Exponentially decaying tenure with spikes at long tenure)
    # Most churn happens early in tenure
    tenure_months = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)
    
    # 4. Service Subscriptions
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == "Yes",
                              np.random.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]),
                              "No phone service")
    
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])
    
    online_security = np.where(internet_service == "No", "No internet service",
                               np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65]))
    tech_support = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.32, 0.68]))
    streaming_tv = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55]))
    
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.22, 0.23])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples, p=[0.34, 0.22, 0.22, 0.22]
    )
    
    # 5. Financial Charges (Calculated with baseline + service additions + noise)
    base_charge = 20.0
    fiber_add = np.where(internet_service == "Fiber optic", 45.0, 0.0)
    dsl_add = np.where(internet_service == "DSL", 25.0, 0.0)
    phone_add = np.where(phone_service == "Yes", 15.0, 0.0)
    security_add = np.where(online_security == "Yes", 10.0, 0.0)
    streaming_add = np.where(streaming_tv == "Yes", 12.0, 0.0)
    noise = np.random.normal(0, 3, size=n_samples)
    
    monthly_charges = np.round(base_charge + fiber_add + dsl_add + phone_add + security_add + streaming_add + noise, 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    # Total charges: roughly monthly_charges * tenure with random deviations
    total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 50, size=n_samples), 2)
    total_charges = np.clip(total_charges, 18.0, 9000.0)
    
    # Customer Service Calls (Poisson distribution - dissatisfied customers call more)
    base_calls = np.random.poisson(lam=1.5, size=n_samples)
    
    # 6. Latent Churn Probability Modeling (Realistic logistic relationship)
    # Higher risk: Month-to-month contract, Fiber optic, High monthly charges, Electronic check, High service calls, Low tenure
    logit = (
        -1.2
        + 1.35 * (contract == "Month-to-month")
        - 1.10 * (contract == "Two year")
        + 0.85 * (internet_service == "Fiber optic")
        + 0.70 * (payment_method == "Electronic check")
        - 0.045 * tenure_months
        + 0.018 * monthly_charges
        + 0.35 * (senior_citizen == 1)
        + 0.30 * base_calls
        - 0.50 * (online_security == "Yes")
        - 0.40 * (tech_support == "Yes")
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    # Customer service calls adjusted by actual dissatisfaction
    customer_service_calls = base_calls + np.where(churn == 1, np.random.poisson(lam=1.8, size=n_samples), 0)
    
    # Satisfaction score (1 to 5): Highly correlated with churn
    satisfaction_score = np.where(
        churn == 1,
        np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.50, 0.30, 0.15, 0.05]),
        np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.05, 0.20, 0.45, 0.30])
    )
    
    # 7. Intentional Subtle Data Leakage Features (To be detected in EDA Checklist Step 7)
    # Feature A: 'cancellation_request_date' (Only populated if customer churned - Post-outcome leakage!)
    cancellation_request = np.where(churn == 1, "Recorded", "None")
    
    # Feature B: 'last_interaction_status' (Proxy for churn outcome - Direct leakage!)
    last_interaction_status = np.where(
        churn == 1,
        np.random.choice(["Account Closed", "Retention Declined", "Final Invoice Dispatched"], size=n_samples, p=[0.70, 0.20, 0.10]),
        np.random.choice(["Active Support", "Bill Query", "Plan Upgrade", "Routine Inquiry"], size=n_samples, p=[0.30, 0.30, 0.20, 0.20])
    )
    
    # Feature C: 'churn_risk_post_exit_code' (Near-perfect deterministic leakage)
    churn_risk_post_exit_code = np.where(churn == 1, 999, 100)
    
    # Assemble raw DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "customer_service_calls": customer_service_calls,
        "satisfaction_score": satisfaction_score,
        "cancellation_request": cancellation_request,           # LEAKAGE FEATURE
        "last_interaction_status": last_interaction_status,     # LEAKAGE FEATURE
        "churn_risk_post_exit_code": churn_risk_post_exit_code, # LEAKAGE FEATURE
        "churn": churn                                          # TARGET
    })
    
    # 8. Inject realistic missing values (MCAR / MAR)
    # Missing values in total_charges (common for brand new customers where tenure == 0 or 1)
    missing_tc_idx = np.random.choice(df.index, size=45, replace=False)
    df.loc[missing_tc_idx, "total_charges"] = np.nan
    
    # Missing values in tech_support (random omission)
    missing_ts_idx = np.random.choice(df.index, size=60, replace=False)
    df.loc[missing_ts_idx, "tech_support"] = np.nan
    
    # Missing values in customer_service_calls (random reporting lapse)
    missing_calls_idx = np.random.choice(df.index, size=35, replace=False)
    df.loc[missing_calls_idx, "customer_service_calls"] = np.nan
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "telecom_churn.csv")
    df = generate_telecom_churn_data(n_samples=3500, random_state=42)
    df.to_csv(csv_path, index=False)
    print(f"Generated realistic telecom churn dataset: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn Distribution:\n{df['churn'].value_counts(normalize=True)}")
    print(f"Missing Values Summary:\n{df.isna().sum()[df.isna().sum() > 0]}")
