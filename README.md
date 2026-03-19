# 🔍 Explainable AI for Financial Fraud Detection & Risk Scoring
 
<div align="center">
 
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-189ABA?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)
![LIME](https://img.shields.io/badge/LIME-Local%20XAI-4CAF50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
 
**MSc Data Analytics | De Montfort University, Leicester**  
*Muhammad Fahad Farooq (P2902499) · Supervisor: Taha Raseem*
 
---
 
> *"It is not enough for a fraud detection system to be accurate — it must be explainable. Under GDPR Article 22 and the EU AI Act 2024, financial institutions are legally required to justify automated decisions that affect customers."*
 
</div>
 
---
 
## 📌 Project Overview
 
Financial fraud costs the UK alone **£572.6 million per year** (FICO European Fraud Map 2024), and global fraud losses reached **$485.6 billion in 2023** (Nasdaq Global Financial Crime Report). Machine learning can detect fraud effectively — but black-box models cannot be legally deployed in regulated financial environments.
 
This project builds a **complete, production-aware fraud detection and explainability framework** that bridges the gap between ML performance and regulatory compliance. It goes far beyond standard classification — implementing dual XAI methods, statistical cross-explainer agreement analysis, calibrated risk scoring, and production-grade drift monitoring.
 
### Why This Matters
 
| Problem | This Project's Solution |
|---|---|
| Fraud is 0.17% of transactions — accuracy is meaningless | PR-AUC and MCC used as primary metrics |
| Black-box ML cannot be legally deployed (GDPR, EU AI Act) | SHAP + LIME provide legally compliant explanations |
| Standard classifiers fail on extreme class imbalance | SMOTE, ADASYN, SMOTETomek compared and justified |
| Probability estimates are poorly calibrated | Platt scaling with reliability diagram validation |
| Models degrade as fraud patterns evolve | PSI concept drift monitoring framework |
| No justifiable decision thresholds | Cost-curve-derived tier boundaries |
 
---
 
## 🏗️ Project Architecture
 
```
XAI Fraud Detection Framework
│
├── 📊 Data & EDA
│   ├── Class imbalance analysis (284,807 transactions)
│   ├── Temporal pattern engineering (Time → Hour of day)
│   ├── Mann-Whitney U + Kolmogorov-Smirnov tests (all features)
│   └── Feature correlation & statistical significance
│
├── ⚙️ Preprocessing & Feature Engineering
│   ├── StandardScaler (Amount_scaled) + log transform (Amount_log)
│   ├── Isolation Forest anomaly score (validated with Mann-Whitney)
│   ├── SMOTE vs ADASYN vs SMOTETomek comparison
│   └── Stratified train/test split (SMOTE on training only — no leakage)
│
├── 🤖 Model Development
│   ├── Logistic Regression (class_weight='balanced')
│   ├── Random Forest (200 estimators, balanced weights)
│   └── XGBoost → RandomizedSearchCV (50 iter, PR-AUC objective)
│
├── 📈 Comprehensive Evaluation (8 Metrics)
│   ├── PR-AUC · ROC-AUC · F1-Score · Precision · Recall
│   ├── MCC · Cohen's Kappa · Brier Score
│   ├── DET curve · Cost-sensitive financial analysis
│   └── 5-fold stratified cross-validation
│
├── 🔍 Explainable AI — SHAP
│   ├── Global: Beeswarm plot + Bar chart (top 15 features)
│   ├── Local: Waterfall plot + Force plot (per transaction)
│   ├── Interaction: Dependence plots (top 2 features)
│   └── Permutation importance (model-agnostic cross-check)
│
├── 🔎 Explainable AI — LIME
│   ├── Local explanation (same transaction as SHAP)
│   ├── LIME vs SHAP Jaccard agreement across 30 transactions
│   └── Statistical agreement reporting (mean ± std)
│
├── 🎯 Fraud Risk Scoring System
│   ├── Platt scaling probability calibration
│   ├── Cost-optimised decision threshold
│   ├── Data-driven tier boundaries (Low/Medium/High/Critical)
│   └── Per-transaction risk report with embedded SHAP explanation
│
└── 🏭 Production Readiness
    ├── Population Stability Index (PSI) drift monitoring
    ├── Feature-level drift alerts with thresholds
    └── Ethical & regulatory discussion (GDPR, EU AI Act 2024)
```
 
---
 
## 📂 Repository Structure
 
```
📦 xai-fraud-detection/
├── 📓 XAI_Fraud_Detection_v4.ipynb    ← Main notebook (run this)
├── 📄 README.md                        ← You are here
├── 📋 requirements.txt                 ← All dependencies
├── 📁 outputs/                         ← Generated plots & reports
│   ├── class_imbalance.png
│   ├── amount_analysis.png
│   ├── temporal_patterns.png
│   ├── isolation_forest.png
│   ├── evaluation_curves.png
│   ├── shap_beeswarm.png
│   ├── shap_waterfall.png
│   ├── shap_force.png
│   ├── shap_dependence.png
│   ├── permutation_importance.png
│   ├── lime_explanation.png
│   ├── lime_shap_agreement.png
│   ├── calibration.png
│   ├── risk_scoring.png
│   ├── cost_analysis.png
│   └── psi_drift.png
└── 📊 creditcard.csv                   ← Dataset (download separately)
```
 
---
 
## 🚀 Quick Start
 
### 1. Clone the repository
 
```bash
git clone https://github.com/yourusername/xai-fraud-detection.git
cd xai-fraud-detection
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Download the dataset
 
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory.
 
> The dataset contains 284,807 anonymised credit card transactions from September 2013 (European cardholders). Features V1–V28 are PCA-transformed for privacy. `Amount` and `Time` are in original scale. `Class` = 1 (fraud), 0 (legitimate).
 
### 4. Run the notebook
 
```bash
jupyter notebook XAI_Fraud_Detection_v4.ipynb
```
 
Then: **Kernel → Restart & Run All**
 
> ⚠️ The RandomizedSearchCV hyperparameter tuning step takes approximately 3–5 minutes. All other cells run in under 1 minute each.
 
---
 
## 📦 Requirements
 
```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
shap>=0.41.0
lime>=0.2.0
scipy>=1.9.0
```
 
Install all at once:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap lime scipy
```
 
---
 
## 📊 Key Results
 
> *Exact values will appear after running the notebook. The structure below shows what is produced.*
 
### Model Performance Comparison
 
| Model | PR-AUC | ROC-AUC | MCC | Recall | Brier |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| **XGBoost (tuned)** | **—** | **—** | **—** | **—** | **—** |
 
> PR-AUC is the primary metric. ROC-AUC inflates results for imbalanced data because it counts true negatives — which are trivially easy when fraud is 0.17% of transactions.
 
### Risk Tier System
 
| Tier | Threshold | Recommended Action |
|---|---|---|
| 🟢 Low | `< low_boundary` | Allow — no action |
| 🟡 Medium | `low_boundary – mid_boundary` | Monitor — batch review |
| 🟠 High | `mid_boundary – high_boundary` | Hold — request authentication |
| 🔴 Critical | `> high_boundary` | Block — immediate analyst review |
 
> Tier boundaries are **derived from the cost optimisation curve**, not hardcoded. This makes them defensible and tied to actual financial impact.
 
---
 
## 🔍 What Makes This Different
 
Most fraud detection projects stop at training a model and printing a classification report. This project goes several layers deeper:
 
### ✅ Methodologically correct imbalance handling
- SMOTE applied **only to training data** — never to the test set, preventing data leakage
- Three oversampling strategies (SMOTE, ADASYN, SMOTETomek) compared before choosing
- `scale_pos_weight` in XGBoost computed from actual training data ratio, not hardcoded
 
### ✅ Right evaluation metrics for fraud
- Standard accuracy is **meaningless** at 0.17% fraud rate (a model predicting "always legitimate" scores 99.83%)
- **PR-AUC** used as primary metric — it is sensitive to the minority class without inflation from true negatives
- **MCC** (Matthews Correlation Coefficient) — the most informative single metric for binary imbalance
- **Brier Score** — tests probability calibration quality, not just discrimination
 
### ✅ Hyperparameter tuning done correctly
- **RandomizedSearchCV** over 9 hyperparameters, 50 random combinations, 5-fold CV
- Optimisation objective: **PR-AUC** — not accuracy, not log-loss
- Best parameters printed and used for all downstream analysis
 
### ✅ Dual XAI with statistical agreement analysis
- **SHAP**: Global (beeswarm + bar) and local (waterfall + force + dependence)
- **LIME**: Local surrogate with 5,000 perturbation samples
- **Cross-explainer agreement**: Jaccard index computed across **30 correctly-predicted fraud transactions** — not just one illustration
- Mean, std, and distribution of agreement scores reported — this is a statistical finding
 
### ✅ Production-grade risk system
- **Platt scaling** calibration — ensures predicted probabilities are reliable, not just discriminative
- **Cost-optimised thresholds** — sweeps 1,000 thresholds, minimises `FN×€120 + FP×€5`
- **Cost-derived tier boundaries** — Low/Medium/High/Critical boundaries come from the cost curve, not guesswork
- **Per-transaction risk report** — embeds calibrated probability + tier + top-5 SHAP features
 
### ✅ PSI drift monitoring (production-ready)
- **Population Stability Index** computed per feature between training distribution and simulated production batch
- PSI < 0.1 = stable, 0.1–0.2 = monitor, > 0.2 = retrain
- Colour-coded monitoring chart with per-feature status flags
- This is the industry-standard metric used by banks to decide when to retrain models
 
---
 
## 🧠 XAI Methods Explained
 
### SHAP (SHapley Additive exPlanations)
Based on **cooperative game theory** (Shapley values), SHAP decomposes each prediction into the contribution of every feature, satisfying three axioms:
- **Efficiency** — contributions sum to prediction minus baseline
- **Consistency** — if a model changes so a feature matters more, its SHAP value increases
- **Null player** — features with no effect get SHAP = 0
 
`TreeExplainer` is used (not `KernelExplainer`) — it is **exact and polynomial-time** for tree-based models like XGBoost.
 
### LIME (Local Interpretable Model-Agnostic Explanations)
LIME perturbs the input around a specific transaction, observes how the model's prediction changes, and fits a **local linear surrogate model** to approximate the decision boundary. Unlike SHAP, LIME is model-agnostic — it treats the model as a black box.
 
### SHAP vs LIME — Why Both?
 
| | SHAP | LIME |
|---|---|---|
| Theoretical basis | Shapley values (game theory) | Local linear approximation |
| Model-specific? | TreeExplainer is XGBoost-specific | Fully model-agnostic |
| Global explanations | Yes | No |
| Stability | High | Can vary between runs |
| Speed | Fast (TreeExplainer) | Slower (perturbation sampling) |
 
Using both and measuring agreement across 30 transactions tests whether explanations are **robust** or artefacts of one method's assumptions.
 
---
 
## ⚖️ Ethical & Regulatory Context
 
### GDPR — Article 22 (Right to Explanation)
Automated transaction blocking constitutes individual automated decision-making. GDPR requires that affected individuals receive a **meaningful explanation**. The SHAP waterfall plots in this project provide exactly that — a ranked list of features and their direction of influence, presentable in plain language to affected customers.
 
### EU AI Act 2024 (Regulation 2024/1689)
Credit card fraud detection is classified as a **High-Risk AI system** under Annex III. Requirements include:
- Technical documentation ✅ (this notebook)
- Human oversight mechanisms ✅ (Critical tier → analyst review)
- Transparency to affected persons ✅ (SHAP-based risk report)
- Ongoing monitoring ✅ (PSI drift framework)
 
### Fairness
PCA transformation partially obscures demographic attributes but cannot guarantee they are unrecoverable from V1–V28 combinations. Any production deployment must undergo **disparate impact testing** across protected characteristics before going live.
 
---
 
## 📚 References
 
| Reference | Relevance |
|---|---|
| Lundberg, S.M. & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS. | SHAP theoretical foundation |
| Ribeiro, M.T. et al. (2016). *"Why should I trust you?" Explaining the predictions of any classifier.* KDD. | LIME theoretical foundation |
| Chawla, N.V. et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR, 16, 321–357. | Oversampling methodology |
| Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD. | Model architecture |
| Dal Pozzolo, A. et al. (2015). *Calibrating probability with undersampling for unbalanced classification.* IEEE SSCI. | Calibration methodology |
| European Commission (2024). *EU Artificial Intelligence Act (2024/1689).* Official Journal of the EU. | Regulatory framework |
| FICO (2024). *European Fraud Map 2024.* | Industry statistics |
| Nasdaq (2024). *Global Financial Crime Report.* | Global fraud scale |
 
---
 
## 🗂️ Notebook Sections
 
| Section | Description |
|---|---|
| **1** | Environment setup & imports |
| **2** | Data loading, EDA, statistical tests |
| **3** | Feature engineering, oversampling comparison, train/test split |
| **4** | Model training + XGBoost RandomizedSearchCV tuning |
| **5** | 8-metric evaluation, DET curve, cost analysis, cross-validation, permutation importance |
| **6** | SHAP — global (beeswarm, bar) + local (waterfall, force, dependence) |
| **7** | LIME — local explanation + multi-transaction agreement analysis |
| **8** | Probability calibration, threshold optimisation, tiered risk scoring |
| **9** | Final model comparison |
| **9.5** | PSI concept drift monitoring |
| **10** | Critical evaluation, limitations, ethics, conclusions |
 
---
 
## 👤 Author
 
**Muhammad Fahad Farooq**  
MSc Data Analytics, De Montfort University, Leicester  
Student ID: P2902499  
Supervisor: Taha Raseem
 
---
 
## 📄 License
 
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
 
The Credit Card Fraud Detection dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
 
---
 
<div align="center">
 
*Built as an MSc Data Analytics dissertation project at De Montfort University, Leicester, UK*
 
</div>
