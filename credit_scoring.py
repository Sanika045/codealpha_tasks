"""
Credit Scoring Model
Predicts individual creditworthiness using financial data.
Uses Logistic Regression, Decision Tree, and Random Forest classifiers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
np.random.seed(42)
N = 2000

def generate_dataset(n=N):
    age = np.random.randint(21, 70, n)
    income = np.random.normal(55000, 20000, n).clip(15000, 200000)
    employment_years = np.random.randint(0, 35, n)
    num_credit_accounts = np.random.randint(1, 15, n)
    total_debt = np.random.normal(20000, 15000, n).clip(0, 120000)
    debt_to_income = total_debt / income
    payment_history_score = np.random.randint(300, 850, n)
    num_late_payments = np.random.randint(0, 20, n)
    credit_utilization = np.random.uniform(0, 1, n)
    num_hard_inquiries = np.random.randint(0, 10, n)
    loan_amount = np.random.normal(15000, 10000, n).clip(1000, 80000)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n)
    savings_balance = np.random.normal(8000, 6000, n).clip(0, 60000)
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n,
                                        p=[0.35, 0.40, 0.18, 0.07])
    employment_status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n,
                                          p=[0.70, 0.20, 0.10])

    # Creditworthiness score (engineered)
    score = (
        0.30 * (payment_history_score / 850) +
        0.20 * (1 - debt_to_income.clip(0, 1)) +
        0.15 * (1 - credit_utilization) +
        0.10 * (employment_years / 35) +
        0.10 * (savings_balance / 60000) +
        0.08 * (income / 200000) +
        0.07 * (1 - num_late_payments / 20) +
        -0.05 * (num_hard_inquiries / 10) +
        np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    creditworthy = (score >= 0.50).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income.round(2),
        'employment_years': employment_years,
        'num_credit_accounts': num_credit_accounts,
        'total_debt': total_debt.round(2),
        'debt_to_income_ratio': debt_to_income.round(4),
        'payment_history_score': payment_history_score,
        'num_late_payments': num_late_payments,
        'credit_utilization': credit_utilization.round(4),
        'num_hard_inquiries': num_hard_inquiries,
        'loan_amount': loan_amount.round(2),
        'loan_term_months': loan_term_months,
        'savings_balance': savings_balance.round(2),
        'education_level': education_level,
        'employment_status': employment_status,
        'creditworthy': creditworthy
    })
    return df

print("=" * 60)
print("       CREDIT SCORING MODEL — FULL PIPELINE")
print("=" * 60)

df = generate_dataset()
print(f"\n✔  Dataset generated: {df.shape[0]} samples, {df.shape[1]} features")
print(f"   Class balance  →  Creditworthy: {df['creditworthy'].sum()} | "
      f"Not: {(df['creditworthy']==0).sum()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[1/5] Feature Engineering …")

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education_level'])
df['employment_encoded'] = le.fit_transform(df['employment_status'])

df['income_per_debt']         = (df['income'] / (df['total_debt'] + 1)).round(4)
df['savings_to_income_ratio'] = (df['savings_balance'] / df['income']).round(4)
df['payment_reliability']     = (1 - df['num_late_payments'] / 20).round(4)
df['credit_age_score']        = (df['employment_years'] * df['num_credit_accounts']).round(2)

FEATURES = [
    'age', 'income', 'employment_years', 'num_credit_accounts',
    'total_debt', 'debt_to_income_ratio', 'payment_history_score',
    'num_late_payments', 'credit_utilization', 'num_hard_inquiries',
    'loan_amount', 'loan_term_months', 'savings_balance',
    'education_encoded', 'employment_encoded',
    'income_per_debt', 'savings_to_income_ratio',
    'payment_reliability', 'credit_age_score'
]
TARGET = 'creditworthy'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[2/5] Training Models …")

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                                        min_samples_leaf=10, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                        min_samples_split=10, min_samples_leaf=5,
                                        random_state=42, n_jobs=-1))
    ])
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                 scoring='roc_auc', n_jobs=-1)
    results[name] = {
        'pipeline': pipeline,
        'y_pred':   y_pred,
        'y_proba':  y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':   recall_score(y_test, y_pred),
        'f1':       f1_score(y_test, y_pred),
        'roc_auc':  roc_auc_score(y_test, y_proba),
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
    }
    print(f"   ✔ {name:22s}  ROC-AUC={results[name]['roc_auc']:.4f}  "
          f"F1={results[name]['f1']:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 4. METRICS SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n[3/5] Metrics Summary")
print(f"\n{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>8} {'ROC-AUC':>9}")
print("-" * 72)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>9.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n★  Best Model: {best_name}  (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
print("\n[4/5] Generating Visualisations …")

PALETTE = {
    'Logistic Regression': '#2563EB',
    'Decision Tree':       '#16A34A',
    'Random Forest':       '#DC2626',
}
BG   = '#0F172A'
CARD = '#1E293B'
TEXT = '#F1F5F9'
MUTED= '#94A3B8'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#334155', 'grid.color': '#1E293B',
    'font.family': 'DejaVu Sans'
})

fig = plt.figure(figsize=(22, 18))
fig.suptitle('CREDIT SCORING MODEL — EVALUATION DASHBOARD',
             fontsize=18, fontweight='bold', color=TEXT, y=0.98)

# ── (A) ROC Curves ──────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_facecolor(CARD)
ax1.plot([0,1],[0,1],'--',color='#475569',lw=1.2,label='Random (AUC=0.50)')
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax1.plot(fpr, tpr, color=PALETTE[name], lw=2.2,
             label=f"{name} ({r['roc_auc']:.3f})")
ax1.set_title('ROC Curves', color=TEXT, fontweight='bold', pad=10)
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=7.5, framealpha=0.2)
ax1.grid(alpha=0.15)

# ── (B) Precision-Recall Curves ─────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.set_facecolor(CARD)
for name, r in results.items():
    p, rc, _ = precision_recall_curve(y_test, r['y_proba'])
    ax2.plot(rc, p, color=PALETTE[name], lw=2.2, label=name)
ax2.set_title('Precision-Recall Curves', color=TEXT, fontweight='bold', pad=10)
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(alpha=0.15)

# ── (C) Metrics Bar Chart ───────────────────
ax3 = fig.add_subplot(3, 3, 3)
ax3.set_facecolor(CARD)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metric_names)); w = 0.25
for i, (name, r) in enumerate(results.items()):
    vals = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['roc_auc']]
    bars = ax3.bar(x + i*w - w, vals, w, label=name,
                   color=PALETTE[name], alpha=0.85, zorder=3)
ax3.set_title('Metrics Comparison', color=TEXT, fontweight='bold', pad=10)
ax3.set_xticks(x); ax3.set_xticklabels(metric_names, fontsize=8)
ax3.set_ylim(0.5, 1.05); ax3.legend(fontsize=7.5, framealpha=0.2)
ax3.grid(axis='y', alpha=0.15, zorder=0)
ax3.axhline(1.0, color='#475569', lw=0.8, ls='--')

# ── (D-F) Confusion Matrices ────────────────
for idx, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(3, 3, 4 + idx)
    ax.set_facecolor(CARD)
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(PALETTE[name], as_cmap=True),
                linewidths=1, linecolor=BG,
                xticklabels=['Not CW', 'Creditworthy'],
                yticklabels=['Not CW', 'Creditworthy'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'Confusion Matrix\n{name}', color=TEXT, fontweight='bold', pad=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# ── (G) Feature Importance (RF) ─────────────
ax7 = fig.add_subplot(3, 3, 7)
ax7.set_facecolor(CARD)
rf_clf = results['Random Forest']['pipeline'].named_steps['clf']
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[-12:]
colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(indices)))
ax7.barh(range(len(indices)), importances[indices], color=colors_imp, alpha=0.9)
ax7.set_yticks(range(len(indices)))
ax7.set_yticklabels([FEATURES[i] for i in indices], fontsize=8)
ax7.set_title('Feature Importance\n(Random Forest — Top 12)', color=TEXT, fontweight='bold', pad=8)
ax7.set_xlabel('Importance Score'); ax7.grid(axis='x', alpha=0.15)

# ── (H) Cross-Val Distribution ──────────────
ax8 = fig.add_subplot(3, 3, 8)
ax8.set_facecolor(CARD)
all_cv_scores = []
all_cv_labels = []
for name, pipeline in models.items():
    s = cross_val_score(pipeline, X_train, y_train, cv=cv,
                        scoring='roc_auc', n_jobs=-1)
    all_cv_scores.append(s)
    all_cv_labels.append(name)

bp = ax8.boxplot(all_cv_scores, patch_artist=True,
                  medianprops=dict(color='white', lw=2))
for patch, name in zip(bp['boxes'], all_cv_labels):
    patch.set_facecolor(PALETTE[name]); patch.set_alpha(0.75)
ax8.set_xticklabels(all_cv_labels, fontsize=8)
ax8.set_title('5-Fold CV ROC-AUC Distribution', color=TEXT, fontweight='bold', pad=8)
ax8.set_ylabel('ROC-AUC'); ax8.grid(axis='y', alpha=0.15)
ax8.set_ylim(0.7, 1.0)

# ── (I) Class Distribution ──────────────────
ax9 = fig.add_subplot(3, 3, 9)
ax9.set_facecolor(CARD)
counts = df['creditworthy'].value_counts()
wedges, texts, autotexts = ax9.pie(
    counts, labels=['Not Creditworthy', 'Creditworthy'],
    colors=['#EF4444', '#22C55E'], autopct='%1.1f%%',
    startangle=90, pctdistance=0.82,
    wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2)
)
for t in texts: t.set_color(TEXT); t.set_fontsize(9)
for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
ax9.set_title('Target Class Distribution', color=TEXT, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('credit_scoring_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("   ✔ Dashboard saved → credit_scoring_dashboard.png")
plt.close()

# ─────────────────────────────────────────────
# 6. CLASSIFICATION REPORTS
# ─────────────────────────────────────────────
print("\n[5/5] Detailed Classification Reports\n")
for name, r in results.items():
    print(f"{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_test, r['y_pred'],
                                 target_names=['Not Creditworthy', 'Creditworthy']))

# ─────────────────────────────────────────────
# 7. SAVE DATASET SAMPLE
# ─────────────────────────────────────────────
df.to_csv('credit_scoring_dataset.csv', index=False)
print("✔  Dataset saved → credit_scoring_dataset.csv")

print("\n" + "=" * 60)
print(f"  PROJECT COMPLETE  |  Best Model: {best_name}")
print(f"  ROC-AUC: {results[best_name]['roc_auc']:.4f}  |  "
      f"F1: {results[best_name]['f1']:.4f}")
print("=" * 60)
