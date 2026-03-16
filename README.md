# 💰 data.org Financial Health Prediction Challenge

> **Zindi Competition** | Hosted by data.org & FinMark Trust | Prize: $1,500 USD

---

## 📌 Overview

This project is a solution to the [data.org Financial Health Prediction Challenge](https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge) on Zindi. The goal is to build a machine learning model that predicts the **Financial Health Index (FHI)** of Small and Medium Enterprises (SMEs) across four Southern African countries: **Eswatini, Lesotho, Zimbabwe, and Malawi**.

The FHI is a composite measure that classifies businesses into **Low**, **Medium**, or **High** financial health across four key dimensions:

| Dimension | Description |
|---|---|
| 💾 Savings & Assets | Business reserves and owned resources |
| 💳 Debt & Repayment Ability | Capacity to service existing obligations |
| 🛡️ Resilience to Shocks | Ability to withstand economic disruption |
| 🏦 Access to Credit & Financial Services | Integration with formal financial systems |

---

## 🎯 Problem Statement

Traditional profit-based metrics fail to capture the full picture of SME wellbeing. The FHI redefines how financial health is measured — beyond profits to **resilience and opportunity**. This challenge supports data-driven policies and inclusive financing strategies for businesses in developing economies.

**Task:** Multi-class classification — predict whether an SME falls into `Low`, `Medium`, or `High` FHI categories.

---

## 📂 Repository Structure

```
Financial-Health-Prediction/
│
├── data/
│   ├── train.csv               # Training data with target FHI labels
│   ├── test.csv                # Test data (no target column)
│   └── sample_submission.csv   # Submission format
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering & cleaning
│   └── 03_modelling.ipynb      # Model training & evaluation
│
├── src/
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── features.py             # Feature engineering functions
│   └── model.py                # Model training & inference
│
├── submissions/
│   └── submission.csv          # Final competition submission
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

The dataset is sourced from socio-economic surveys and business records across Southern Africa. Features include:

- **Traded commodities** — types and volume of goods traded
- **Export & import activity** — cross-border trade indicators
- **Demographics** — owner age, gender, education level
- **Firm size** — number of employees, years in operation
- **Location** — urban/rural classification, country

> ⚠️ Data is provided exclusively through Zindi and must not be redistributed. Download it from the [competition page](https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge).

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Financial-Health-Prediction.git
cd Financial-Health-Prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add data
Place the downloaded Zindi data files into the `data/` directory.

---

## 🚀 Usage

### Run EDA
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Train the model
```bash
python src/model.py --train
```

### Generate submission
```bash
python src/model.py --predict --output submissions/submission.csv
```

---

## 🧠 Approach

1. **Exploratory Data Analysis** — Understanding class imbalance, feature distributions, and correlations across countries.
2. **Feature Engineering** — Encoding categorical variables, creating interaction features, handling missing values.
3. **Modelling** — Experimenting with gradient boosted trees (XGBoost / LightGBM), ensemble methods, and cross-validation strategies.
4. **Evaluation** — Optimising for the competition metric on a held-out validation set before final submission.

---

## 📊 Evaluation Metric

Submissions are evaluated using the metric specified on the Zindi competition page (log loss / accuracy — confirm on the leaderboard page).

- **Public Leaderboard:** ~20% of test data
- **Private Leaderboard:** remaining 80% of test data

---

## 📋 Competition Rules Summary

- Maximum **10 submissions per day**
- Teams of up to **4 members**
- Only **publicly available, open-source packages** permitted
- Top 10 finalists must submit reproducible code within **48 hours** of being contacted
- Data may not be shared outside the competition platform

---

## 🏆 Prize

**$1,500 USD** — paid via bank transfer or PayPal.

---

## 📚 Resources

- [Zindi Competition Page](https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge)
- [data.org Platform](https://data.org)
- [FinMark Trust](https://finmark.org.za)
- [Capacity Accelerator Network (CAN)](https://data.org/networks/can/) — free accreditation courses for financial inclusion data science

---

## 🤝 Acknowledgements

Challenge hosted by **data.org** and **FinMark Trust**, with data sourced from FinScope surveys across Southern Africa.

---

## 📄 License

This project is for competition purposes only. All data is the property of Zindi and the challenge host — refer to the [Zindi competition rules](https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge) for usage terms.
