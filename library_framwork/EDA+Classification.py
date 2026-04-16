# ============================================================
#  MINH TRƯỜNG — EDA + CLASSIFICATION  |  Hà Nội AQI 2022-2025
#  Yêu cầu: 5 biểu đồ EDA, nhận xét phân tích, so sánh 3 classifier,
#            confusion matrix từng class, lưu best model XGBoost
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from numpy.polynomial.polynomial import polyfit
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CẤU HÌNH CHUNG
# ─────────────────────────────────────────────────────────────
import os as _os
_BASE     = _os.path.dirname(_os.path.abspath(__file__))
DATA_PATH = _os.path.join(_BASE, '..', 'clean', 'hanoi_aqi_cleaned.csv')
OUT       = _BASE + _os.sep   # output cùng thư mục với file .py

PALETTE = {
    'Good':                           '#2ecc71',
    'Moderate':                       '#f1c40f',
    'Unhealthy for sensitive groups': '#e67e22',
    'Unhealthy':                      '#e74c3c',
    'Very Unhealthy':                 '#9b59b6',
    'Hazardous':                      '#7f1d1d',
}
CAT_ORDER    = list(PALETTE.keys())
SEASON_MAP   = {0: 'Spring', 1: 'Summer', 2: 'Autumn', 3: 'Winter'}
SEASON_ORDER = ['Spring', 'Summer', 'Autumn', 'Winter']
SEASON_CLR   = {'Spring': '#27ae60', 'Summer': '#f39c12',
                'Autumn': '#e67e22', 'Winter': '#3498db'}

FEAT_COLS = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
             'clouds', 'precipitation', 'pressure', 'relative_humidity',
             'temperature', 'wind_speed',
             'hour', 'month', 'season', 'is_weekend', 'day_of_week']

sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({'figure.dpi': 150, 'axes.titlepad': 10})

# ─────────────────────────────────────────────────────────────
#  ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['aqi_category'] = pd.Categorical(df['aqi_category'],
                                    categories=CAT_ORDER, ordered=True)
df['season_name'] = df['season'].map(SEASON_MAP)
df['ym'] = pd.to_datetime(
    df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
)

print(f"Dữ liệu: {df.shape[0]:,} hàng × {df.shape[1]} cột")
print("Phân bố AQI Category:\n", df['aqi_category'].value_counts(), "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  EDA 1 — Phân phối AQI trung bình theo giờ trong ngày
# ═══════════════════════════════════════════════════════════════════════════
print("── EDA 1: AQI theo giờ ─────────────────────────────────────────────")
hourly = df.groupby('hour')['aqi'].mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(hourly.index, hourly.values,
        color='#e74c3c', lw=2.5, marker='o', ms=5, zorder=3)
ax.fill_between(hourly.index, hourly.values, alpha=0.15, color='#e74c3c')
ax.set_xticks(range(24))
ax.set_xlabel('Giờ trong ngày', fontsize=12)
ax.set_ylabel('AQI trung bình', fontsize=12)
ax.set_title('Phân phối AQI trung bình theo giờ trong ngày — Hà Nội 2022–2025',
             fontsize=13, fontweight='bold')

peak_h = int(hourly.idxmax())
best_h = int(hourly.idxmin())
ax.annotate(f'Đỉnh ô nhiễm: {hourly[peak_h]:.1f} lúc {peak_h}h',
            xy=(peak_h, hourly[peak_h]),
            xytext=(peak_h + 1.8, hourly[peak_h] + 3),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#ffe6e6', ec='#e74c3c', alpha=0.8))
ax.annotate(f'Sạch nhất: {hourly[best_h]:.1f} lúc {best_h}h',
            xy=(best_h, hourly[best_h]),
            xytext=(best_h + 1.8, hourly[best_h] - 7),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#e8f8f5', ec='#27ae60', alpha=0.8))
plt.tight_layout()
fig.savefig(OUT + 'eda1_aqi_by_hour.png')
plt.close()
print(f"  → Giờ ô nhiễm nhất: {peak_h}h  (AQI={hourly[peak_h]:.1f})")
print(f"  → Giờ sạch nhất:    {best_h}h  (AQI={hourly[best_h]:.1f})")
print()


# ═══════════════════════════════════════════════════════════════════════════
#  EDA 2 — Boxplot AQI theo 4 mùa
# ═══════════════════════════════════════════════════════════════════════════
print("── EDA 2: Boxplot theo mùa ──────────────────────────────────────────")
season_stats = df.groupby('season_name')['aqi'].agg(['mean', 'median'])
worst_season = season_stats['mean'].idxmax()

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=df, x='season_name', y='aqi', order=SEASON_ORDER,
            palette=SEASON_CLR, width=0.5, linewidth=1.5,
            flierprops=dict(marker='o', markersize=2, alpha=0.3), ax=ax)
ax.set_xlabel('Mùa', fontsize=12)
ax.set_ylabel('AQI', fontsize=12)
ax.set_title('Phân phối AQI theo 4 mùa — Hà Nội 2022–2025',
             fontsize=13, fontweight='bold')

med = season_stats.loc[worst_season, 'median']
idx = SEASON_ORDER.index(worst_season)
ax.annotate(f'AQI cao nhất:\n{worst_season} (median={med:.0f})',
            xy=(idx, med),
            xytext=(idx + 0.4, med + 20),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='#fdecea', ec='#e74c3c', alpha=0.9))

for i, s in enumerate(SEASON_ORDER):
    m = season_stats.loc[s, 'mean']
    ax.text(i, ax.get_ylim()[0] + 5, f'avg={m:.0f}',
            ha='center', fontsize=8.5, color='#333')
plt.tight_layout()
fig.savefig(OUT + 'eda2_boxplot_season.png')
plt.close()
print(f"  → Mùa AQI cao nhất: {worst_season}")
for s in SEASON_ORDER:
    print(f"     {s}: mean={season_stats.loc[s,'mean']:.1f}  median={season_stats.loc[s,'median']:.1f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
#  EDA 3 — Heatmap tương quan 13 features
# ═══════════════════════════════════════════════════════════════════════════
print("── EDA 3: Correlation Heatmap ───────────────────────────────────────")
FEAT13 = ['aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
          'clouds', 'precipitation', 'pressure', 'relative_humidity',
          'temperature', 'wind_speed']
corr = df[FEAT13].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, linewidths=0.5, annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Heatmap tương quan 13 features — Hà Nội AQI',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT + 'eda3_correlation_heatmap.png')
plt.close()

# In top correlations với AQI
aqi_corr = corr['aqi'].drop('aqi').abs().sort_values(ascending=False)
print("  → Top 5 features tương quan với AQI:")
for feat, val in aqi_corr.head(5).items():
    print(f"     {feat}: {corr['aqi'][feat]:+.3f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
#  EDA 4 — Trend AQI trung bình theo tháng 2022–2025
# ═══════════════════════════════════════════════════════════════════════════
print("── EDA 4: Trend theo tháng ──────────────────────────────────────────")
monthly = df.groupby('ym')['aqi'].mean().reset_index()
x_num   = np.arange(len(monthly))
c       = polyfit(x_num, monthly['aqi'].values, 1)
trend   = c[0] + c[1] * x_num
direction = '↑ TĂNG' if c[1] > 0 else '↓ GIẢM'

fig, ax = plt.subplots(figsize=(14, 5))
# Nền từng năm
for yr, col in zip([2022, 2023, 2024, 2025],
                   ['#fdfefe', '#eaf4fb', '#fdfefe', '#fef9e7']):
    ax.axvspan(pd.Timestamp(f'{yr}-01-01'),
               pd.Timestamp(f'{yr}-12-31'),
               alpha=0.35, color=col, lw=0)
    ax.text(pd.Timestamp(f'{yr}-07-01'), ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 60,
            str(yr), ha='center', fontsize=9, color='#aaa')

ax.plot(monthly['ym'], monthly['aqi'],
        color='#2980b9', lw=2, marker='o', ms=4, zorder=3, label='AQI tháng')
ax.fill_between(monthly['ym'], monthly['aqi'], alpha=0.12, color='#2980b9')
ax.plot(monthly['ym'], trend, '--', color='#e74c3c', lw=2,
        label=f'Xu hướng tổng thể ({direction}, slope={c[1]:.2f}/tháng)')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
plt.xticks(rotation=45, ha='right')
ax.set_xlabel('Tháng', fontsize=12)
ax.set_ylabel('AQI trung bình', fontsize=12)
ax.set_title('Trend AQI trung bình theo tháng — Hà Nội 2022–2025',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(OUT + 'eda4_monthly_trend.png')
plt.close()

yearly = df.groupby('year')['aqi'].mean()
print(f"  → Xu hướng: {direction}  (slope={c[1]:.4f}/tháng)")
print("  → AQI trung bình từng năm:")
for yr, v in yearly.items():
    print(f"     {yr}: {v:.1f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
#  EDA 5 — Tỷ lệ 6 mức AQI Category
# ═══════════════════════════════════════════════════════════════════════════
print("── EDA 5: Tỷ lệ AQI Category ────────────────────────────────────────")
cat_counts = df['aqi_category'].value_counts().reindex(CAT_ORDER)
colors     = [PALETTE[c] for c in CAT_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Pie chart
wedges, _, autotexts = axes[0].pie(
    cat_counts, labels=None, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    pctdistance=0.78)
for at in autotexts:
    at.set_fontsize(9)
axes[0].legend(wedges,
               [f'{c}  ({v:,})' for c, v in zip(CAT_ORDER, cat_counts)],
               loc='lower center', bbox_to_anchor=(0.5, -0.15),
               fontsize=8, ncol=2)
axes[0].set_title('Tỷ lệ 6 mức AQI Category', fontsize=12, fontweight='bold')

# Bar chart
bars = axes[1].bar(range(len(CAT_ORDER)), cat_counts.values,
                   color=colors, edgecolor='white', linewidth=0.8)
axes[1].set_xticks(range(len(CAT_ORDER)))
axes[1].set_xticklabels(
    [c.replace(' for sensitive groups', '\nfor sensitive groups')
     for c in CAT_ORDER], fontsize=8.5)
axes[1].set_ylabel('Số bản ghi', fontsize=11)
axes[1].set_title('Số lượng theo 6 mức AQI', fontsize=12, fontweight='bold')
for bar, v in zip(bars, cat_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 80,
                 f'{v:,}\n({v/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
fig.savefig(OUT + 'eda5_category_distribution.png')
plt.close()

print("  → Phân bố AQI Category:")
for cat, v in cat_counts.items():
    print(f"     {cat:<38}: {v:>6,}  ({v/len(df)*100:.1f}%)")
print()


# ═══════════════════════════════════════════════════════════════════════════
#  NHẬN XÉT PHÂN TÍCH TỔNG HỢP
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  NHẬN XÉT PHÂN TÍCH EDA")
print("=" * 65)
print(f"  1. Giờ ô nhiễm nhất : {peak_h}h (AQI={hourly[peak_h]:.1f}) — ban đêm")
print(f"     Giờ sạch nhất     : {best_h}h (AQI={hourly[best_h]:.1f}) — chiều")
print(f"  2. Mùa AQI cao nhất : {worst_season} (avg AQI={season_stats.loc[worst_season,'mean']:.1f})")
print(f"  3. Xu hướng 4 năm   : {direction} (slope={c[1]:.4f}/tháng)")
print(f"     2022→{yearly.index[-1]}: {yearly.iloc[0]:.1f} → {yearly.iloc[-1]:.1f}")
print(f"  4. Class mất cân bằng: Good {cat_counts['Good']/len(df)*100:.1f}%,"
      f" Hazardous {cat_counts['Hazardous']/len(df)*100:.1f}%")
print("     → Áp dụng SMOTE để cân bằng trước khi train")
print("=" * 65)
print()


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
print("── Chuẩn bị dữ liệu classification ─────────────────────────────────")
X  = df[FEAT_COLS].fillna(df[FEAT_COLS].median())
le = LabelEncoder()
y  = le.fit_transform(df['aqi_category'])
print("  Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  Trước SMOTE — Train: {len(X_train):,}  |  Test: {len(X_test):,}")
after = pd.Series(y_train_sm).value_counts().sort_index()
print(f"  Sau  SMOTE — Train: {len(X_train_sm):,}")
print(f"  Phân bố sau SMOTE: { {le.classes_[k]: v for k, v in after.items()} }")

# Scale (dùng cho LR)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sm)
X_test_sc   = scaler.transform(X_test)

y_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
results = {}

# ── 1. Logistic Regression ─────────────────────────────────────────────────
print("\n── Logistic Regression ──────────────────────────────────────────────")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train_sm)
y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)
f1_lr  = f1_score(y_test, y_pred_lr, average='macro')
auc_lr = roc_auc_score(y_bin, y_prob_lr, average='macro', multi_class='ovr')
results['Logistic\nRegression'] = {
    'f1_macro': f1_lr, 'roc_auc': auc_lr,
    'pred': y_pred_lr, 'prob': y_prob_lr
}
print(f"  F1-macro = {f1_lr:.4f}   ROC-AUC = {auc_lr:.4f}")
print(classification_report(y_test, y_pred_lr,
                             target_names=le.classes_, digits=4))

# ── 2. Random Forest ───────────────────────────────────────────────────────
print("── Random Forest ────────────────────────────────────────────────────")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)
f1_rf  = f1_score(y_test, y_pred_rf, average='macro')
auc_rf = roc_auc_score(y_bin, y_prob_rf, average='macro', multi_class='ovr')
results['Random\nForest'] = {
    'f1_macro': f1_rf, 'roc_auc': auc_rf,
    'pred': y_pred_rf, 'prob': y_prob_rf
}
print(f"  F1-macro = {f1_rf:.4f}   ROC-AUC = {auc_rf:.4f}")
print(classification_report(y_test, y_pred_rf,
                             target_names=le.classes_, digits=4))

# ── 3. XGBoost ─────────────────────────────────────────────────────────────
print("── XGBoost ──────────────────────────────────────────────────────────")
xgb = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                     eval_metric='mlogloss', random_state=42,
                     n_jobs=-1, verbosity=0)
xgb.fit(X_train_sm, y_train_sm)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)
f1_xgb  = f1_score(y_test, y_pred_xgb, average='macro')
auc_xgb = roc_auc_score(y_bin, y_prob_xgb, average='macro', multi_class='ovr')
results['XGBoost'] = {
    'f1_macro': f1_xgb, 'roc_auc': auc_xgb,
    'pred': y_pred_xgb, 'prob': y_prob_xgb
}
print(f"  F1-macro = {f1_xgb:.4f}   ROC-AUC = {auc_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb,
                             target_names=le.classes_, digits=4))

# Lưu best model XGBoost
joblib.dump(
    {'model': xgb, 'scaler': scaler,
     'label_encoder': le, 'features': FEAT_COLS},
    OUT + 'xgboost_best_model.pkl'
)
print("  → xgboost_best_model.pkl đã lưu (dùng cho SHAP)")


# ═══════════════════════════════════════════════════════════════════════════
#  BẢNG SO SÁNH 3 CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
model_names  = list(results.keys())
f1s          = [results[m]['f1_macro'] for m in model_names]
aucs         = [results[m]['roc_auc']  for m in model_names]
best_idx     = int(np.argmax(f1s))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# ── Bar: F1-macro ──
bar_colors = ['#95a5a6'] * 3
bar_colors[best_idx] = '#27ae60'
b1 = axes[0].bar(model_names, f1s, color=bar_colors,
                  edgecolor='white', linewidth=0.8, width=0.5)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('F1-macro Score', fontsize=12)
axes[0].set_title('F1-macro — 3 Classifiers', fontsize=12, fontweight='bold')
for bar, v in zip(b1, f1s):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
axes[0].axhline(0.9, ls='--', color='#e74c3c', lw=1, label='0.9 threshold')
axes[0].legend(fontsize=9)

# ── Bar: ROC-AUC ──
bar_colors2 = ['#95a5a6'] * 3
bar_colors2[best_idx] = '#2980b9'
b2 = axes[1].bar(model_names, aucs, color=bar_colors2,
                  edgecolor='white', linewidth=0.8, width=0.5)
axes[1].set_ylim(0.95, 1.005)
axes[1].set_ylabel('ROC-AUC Score', fontsize=12)
axes[1].set_title('ROC-AUC — 3 Classifiers', fontsize=12, fontweight='bold')
for bar, v in zip(b2, aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.0003,
                 f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

fig.suptitle('So sánh 3 Classifier — F1-macro & ROC-AUC\n'
             f'(Best model: {model_names[best_idx].replace(chr(10)," ")} — highlight xanh lá)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT + 'clf1_comparison_chart.png', bbox_inches='tight')
plt.close()
print("\n  → clf1_comparison_chart.png saved")


# ═══════════════════════════════════════════════════════════════════════════
#  F1 PER-CLASS — 3 models
# ═══════════════════════════════════════════════════════════════════════════
short_labels = ['Good', 'Hazardous', 'Moderate',
                'Unhealthy', 'USG', 'Very\nUnhealthy']
x = np.arange(len(le.classes_))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 5))
for i, (mname, res) in enumerate(results.items()):
    report = classification_report(y_test, res['pred'],
                                   output_dict=True, zero_division=0)
    f1_per = [report[str(k)]['f1-score'] for k in range(len(le.classes_))]
    ax.bar(x + i * width, f1_per, width,
           label=mname.replace('\n', ' '),
           color=['#3498db', '#27ae60', '#e67e22'][i],
           edgecolor='white', linewidth=0.6)

ax.set_xticks(x + width)
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel('F1-score', fontsize=12)
ax.set_title('F1-score từng class — So sánh 3 Classifiers',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0.9, ls='--', color='#e74c3c', lw=1, alpha=0.6, label='0.9 line')
plt.tight_layout()
fig.savefig(OUT + 'clf2_f1_per_class.png', bbox_inches='tight')
plt.close()
print("  → clf2_f1_per_class.png saved")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX — từng model (% dạng heatmap)
# ═══════════════════════════════════════════════════════════════════════════
cm_labels = [c.replace(' for sensitive groups', '\n(USG)') for c in le.classes_]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (mname, res) in zip(axes, results.items()):
    cm     = confusion_matrix(y_test, res['pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    g = sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=cm_labels, yticklabels=cm_labels,
                    linewidths=0.5, cbar_kws={'label': '%'}, ax=ax,
                    annot_kws={'size': 8})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7.5, rotation=30, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7.5, rotation=0)
    ax.set_title(f'{mname.replace(chr(10)," ")}\nF1={res["f1_macro"]:.3f}  AUC={res["roc_auc"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

fig.suptitle('Confusion Matrix (%) từng class — 3 Classifiers',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT + 'clf3_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("  → clf3_confusion_matrices.png saved")


# ═══════════════════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  KẾT QUẢ CUỐI")
print("=" * 65)
print(f"  {'Model':<25} {'F1-macro':>10} {'ROC-AUC':>10}")
print("  " + "-" * 48)
for m, f1, auc in zip(model_names, f1s, aucs):
    flag = " ← BEST" if f1 == max(f1s) else ""
    print(f"  {m.replace(chr(10),' '):<25} {f1:>10.4f} {auc:>10.4f}{flag}")
print("=" * 65)
print("\nFiles đã lưu:")
files = [
    'eda1_aqi_by_hour.png',
    'eda2_boxplot_season.png',
    'eda3_correlation_heatmap.png',
    'eda4_monthly_trend.png',
    'eda5_category_distribution.png',
    'clf1_comparison_chart.png',
    'clf2_f1_per_class.png',
    'clf3_confusion_matrices.png',
    'xgboost_best_model.pkl',
]
for f in files:
    print(f"  ✓ {f}")
print("\n✅ Hoàn thành toàn bộ nhiệm vụ Minh Trường!")
