import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data():
    train = pd.read_csv('Train.csv')
    test = pd.read_csv('Test.csv')
    return train, test

# ============================================================
# TARGET ENCODING HELPER
# ============================================================
def add_target_encoding(train_df, test_df, cat_cols, target):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target_dummies = pd.get_dummies(target)
    class_names = target_dummies.columns
    
    for col in cat_cols:
        for cls in class_names:
            col_name = f'{col}_te_{cls}'
            train_encoded[col_name] = 0
            test_encoded[col_name] = 0
            global_mean = target_dummies[cls].mean()
            summary = target_dummies[cls].groupby(train_df[col]).mean()
            test_encoded[col_name] = test_df[col].map(summary).fillna(global_mean)
            for tr_idx, val_idx in kf.split(train_df, target):
                fold_summary = target_dummies[cls].iloc[tr_idx].groupby(train_df[col].iloc[tr_idx]).mean()
                train_encoded.loc[train_encoded.index[val_idx], col_name] = train_df[col].iloc[val_idx].map(fold_summary).fillna(global_mean)
                
    return train_encoded, test_encoded

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
def add_features(df):
    df = df.copy()
    df['country'] = df['country'].str.strip().str.title()
    
    # NEW: Eswatini Focus (65% of High class cases are here)
    df['is_eswatini'] = (df['country'] == 'Eswatini').astype(int)
    
    # Binary Cleaning
    binary_cols = [
        'compliance_income_tax', 'perception_insurance_important',
        'keeps_financial_records', 'covid_essential_service', 
        'attitude_satisfied_with_achievement', 'attitude_more_successful_next_year',
        'problem_sourcing_money', 'marketing_word_of_mouth', 'motivation_make_more_money'
    ]
    
    def clean_binary(val):
        if pd.isna(val): return "Unknown"
        val = str(val).lower()
        if 'yes' in val: return "Yes"
        if "don't know" in val or "don’t know" in val: return "Unknown"
        return "No"

    for col in binary_cols:
        if col in df.columns:
            df[f'{col}_clean'] = df[col].apply(clean_binary)

    # High-Impact Interactions & Indicators
    df['country_sex'] = df['country'] + "_" + df['owner_sex'].fillna("Unknown")
    df['country_records'] = df['country'] + "_" + df['keeps_financial_records'].fillna("Unknown").astype(str)
    
    # Financial Literacy Proxy
    df['fin_literacy_proxy'] = ((df['keeps_financial_records'].astype(str).str.contains('Yes', case=False)).astype(int) + 
                                (df['compliance_income_tax_clean'] == "Yes").astype(int) + 
                                (df['has_internet_banking'].astype(str).str.contains('Have now', case=False)).astype(int))

    # Stability Profile
    df['stable_profile'] = ((df['attitude_worried_shutdown'] == "No").astype(int) + 
                            (df['attitude_stable_business_environment'] == "Yes").astype(int))

    # Lender history
    df['had_informal_lender_hx'] = df['uses_informal_lender'].astype(str).str.contains('Used to have', case=False).astype(int)

    # Multi-state Cleaning
    multi_cols = [
        'has_credit_card', 'has_debit_card', 'has_internet_banking',
        'has_loan_account', 'has_mobile_money', 'has_insurance',
        'medical_insurance', 'funeral_insurance', 'motor_vehicle_insurance'
    ]

    def clean_multi(val):
        if pd.isna(val): return "Unknown"
        val = str(val).lower().replace('’', "'").replace('don?t', "don't")
        if 'have now' in val: return "HaveNow"
        return "NotHave"

    for col in multi_cols:
        if col in df.columns:
            df[f'{col}_clean'] = df[col].apply(clean_multi)

    # Financial access aggregate
    fin_access_cols = [f'{c}_clean' for c in multi_cols if f'{c}_clean' in df.columns]
    df['fin_access_score'] = (df[fin_access_cols] == "HaveNow").sum(axis=1)

    # Standardize expenses
    df['expense_cycle_likely_monthly'] = (df['business_expenses'] * 12 <= df['business_turnover'] * 1.5).astype(int)
    df['annual_expenses_est'] = df['business_expenses']
    df.loc[df['expense_cycle_likely_monthly'] == 1, 'annual_expenses_est'] = df['business_expenses'] * 12
    
    df['profit_proxy'] = df['business_turnover'] - df['annual_expenses_est']
    df['expense_ratio'] = df['annual_expenses_est'] / (df['business_turnover'] + 1)
    df['profit_margin'] = df['profit_proxy'] / (df['business_turnover'] + 1)
    
    # Log transforms
    skew_cols = ['personal_income', 'business_turnover', 'annual_expenses_est', 'profit_proxy']
    for col in skew_cols:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    return df

# ============================================================
# STEP 3: PREPROCESSING
# ============================================================
def preprocess(train_df, test_df):
    target_col = 'Target'
    y = train_df[target_col]
    test_ids = test_df['ID']
    
    train_feat = add_features(train_df.drop(columns=['ID', target_col]))
    test_feat = add_features(test_df.drop(columns=['ID']))
    
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)
    
    # Target Encoding
    te_cols = ['country', 'country_sex', 'country_records']
    train_feat, test_feat = add_target_encoding(train_feat, test_feat, te_cols, y_enc)
    
    cat_cols = train_feat.select_dtypes(include=['object']).columns.tolist()
    num_cols = train_feat.select_dtypes(include=['number']).columns.tolist()
    
    for col in num_cols:
        med = train_feat[col].median()
        train_feat[col] = train_feat[col].fillna(med)
        test_feat[col] = test_feat[col].fillna(med)
        
    combined_cat = pd.concat([train_feat[cat_cols], test_feat[cat_cols]], axis=0).fillna("Missing").astype(str)
    for col in cat_cols:
        le = LabelEncoder()
        combined_cat[col] = le.fit_transform(combined_cat[col])
        
    train_enc = train_feat[num_cols].copy()
    test_enc = test_feat[num_cols].copy()
    train_enc[cat_cols] = combined_cat.iloc[:len(train_feat)].values
    test_enc[cat_cols] = combined_cat.iloc[len(train_feat):].values
    
    return train_enc, test_enc, y_enc, le_target, test_ids, cat_cols

# ============================================================
# STEP 4: THRESHOLD OPTIMIZATION
# ============================================================
def optimize_thresholds(y_true, y_probs):
    def objective(thresholds):
        preds = []
        for i in range(len(y_probs)):
            adj_probs = y_probs[i] * thresholds
            preds.append(np.argmax(adj_probs))
        return -f1_score(y_true, preds, average='macro')

    res = minimize(objective, [1.0, 1.0, 1.0], method='Nelder-Mead')
    return res.x

# ============================================================
# STEP 5: TRAIN & STACKED ENSEMBLE
# ============================================================
def train_and_evaluate():
    train_df, test_df = load_data()
    X, X_test, y, le_target, test_ids, cat_cols = preprocess(train_df, test_df)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_count = 4
    oof_probs = np.zeros((len(X), 3 * models_count))
    test_probs = np.zeros((len(X_test), 3 * models_count))
    
    sampler = SMOTETomek(random_state=42)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # Apply SMOTETomek only on training index
        X_tr_sm, y_tr_sm = sampler.fit_resample(X_tr, y_tr)
        
        # 1. CatBoost (still using balanced weights for extra safety)
        m1 = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6, auto_class_weights='Balanced', 
                                silent=True, random_seed=42, cat_features=cat_cols)
        m1.fit(X_tr_sm, y_tr_sm, eval_set=(X_val, y_val), early_stopping_rounds=100)
        oof_probs[val_idx, 0:3] = m1.predict_proba(X_val)
        test_probs[:, 0:3] += m1.predict_proba(X_test) / 5
        
        # 2. Random Forest (Balanced)
        m2 = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        m2.fit(X_tr_sm, y_tr_sm)
        oof_probs[val_idx, 3:6] = m2.predict_proba(X_val)
        test_probs[:, 3:6] += m2.predict_proba(X_test) / 5
        
        # 3. Extra Trees (Balanced)
        m3 = ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        m3.fit(X_tr_sm, y_tr_sm)
        oof_probs[val_idx, 6:9] = m3.predict_proba(X_val)
        test_probs[:, 6:9] += m3.predict_proba(X_test) / 5
        
        # 4. Hist Gradient Boosting
        m4 = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_depth=7, random_state=42)
        m4.fit(X_tr_sm, y_tr_sm)
        oof_probs[val_idx, 9:12] = m4.predict_proba(X_val)
        test_probs[:, 9:12] += m4.predict_proba(X_test) / 5
        
        print(f"Fold {fold+1} complete.", flush=True)

    # Meta-Learner Layer 2
    # Standardize OOF probs for better meta-model convergence
    scaler = StandardScaler()
    X_meta = scaler.fit_transform(oof_probs)
    X_test_meta = scaler.transform(test_probs)
    
    meta_model = LogisticRegression(C=0.5, multi_class='multinomial', solver='lbfgs', max_iter=2000, random_state=42)
    meta_model.fit(X_meta, y)
    
    final_oof_probs = meta_model.predict_proba(X_meta)
    final_test_probs = meta_model.predict_proba(X_test_meta)
    
    # Global Threshold Optimization for Macro F1
    best_thresholds = optimize_thresholds(y, final_oof_probs)
    print(f"Optimal Thresholds: {best_thresholds}", flush=True)
    
    final_oof_preds = [np.argmax(p * best_thresholds) for p in final_oof_probs]
    f1 = f1_score(y, final_oof_preds, average='macro')
    print(f"\nFinal REFINED Macro F1: {f1:.4f}", flush=True)
    print(classification_report(y, final_oof_preds, target_names=le_target.classes_), flush=True)
    
    # Save submission v3
    final_test_preds = [np.argmax(p * best_thresholds) for p in final_test_probs]
    submission = pd.DataFrame({'ID': test_ids, 'Target': le_target.inverse_transform(final_test_preds)})
    submission.to_csv('submission_final_v3.csv', index=False)
    
    # Save confusion matrix v3
    cm = confusion_matrix(y, final_oof_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title(f'Refined Confusion Matrix (Macro F1: {f1:.4f})')
    plt.savefig('confusion_matrix_final_v3.png')

if __name__ == '__main__':
    train_and_evaluate()
