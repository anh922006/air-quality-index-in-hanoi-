"""
Dự báo AQI Hà Nội – XGBoost tự cài đặt từ đầu (không dùng thư viện xgboost)
Train: 2022–2024 | Test: 2025
Chỉ sử dụng: numpy, pandas, matplotlib, sqlalchemy, joblib
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

load_dotenv()
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
df = pd.read_sql("SELECT * FROM aqi_data", con=engine)
print(f"Đã đọc {len(df):,} hàng từ MySQL")

df.columns = df.columns.str.strip().str.lower()
df["local_time"] = pd.to_datetime(df["local_time"])
df = df.sort_values("local_time").reset_index(drop=True)

class XGBNode:
    """Một node trong cây quyết định của XGBoost."""

    def __init__(self):
        self.is_leaf    = False
        self.value      = 0.0          # leaf weight
        self.feature    = None         # split feature index
        self.threshold  = None         # split threshold
        self.left       = None
        self.right      = None


class XGBTree:
    """
    Một cây hồi quy dùng thuật toán XGBoost (exact greedy split).
    Loss: MSE  →  gradient g = y_pred - y_true,  hessian h = 1
    """

    def __init__(self, max_depth=4, min_child_weight=1,
                 reg_lambda=1.0, gamma=0.0):
        self.max_depth        = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda       = reg_lambda   # L2 regularisation
        self.gamma            = gamma        # min gain to split
        self.root             = None

    # ── leaf weight ──────────────────────────────────
    def _leaf_weight(self, g, h):
        """w* = -sum(g) / (sum(h) + lambda)"""
        return -g.sum() / (h.sum() + self.reg_lambda)

    # ── gain of a split ──────────────────────────────
    def _gain(self, g_l, h_l, g_r, h_r, g, h):
        """
        Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
        """
        def score(gg, hh):
            return gg**2 / (hh + self.reg_lambda)

        return 0.5 * (score(g_l.sum(), h_l.sum())
                      + score(g_r.sum(), h_r.sum())
                      - score(g.sum(), h.sum())) - self.gamma

    # ── best split ───────────────────────────────────
    def _best_split(self, X, g, h):
        n, n_feat = X.shape
        best_gain  = -np.inf
        best_feat  = None
        best_thresh = None

        for feat in range(n_feat):
            values   = X[:, feat]
            sorted_i = np.argsort(values)
            sv, sg, sh = values[sorted_i], g[sorted_i], h[sorted_i]

            # prefix sums for efficiency
            g_cum = np.cumsum(sg)
            h_cum = np.cumsum(sh)
            G, H  = g_cum[-1], h_cum[-1]

            for i in range(1, n):
                # avoid duplicate thresholds
                if sv[i] == sv[i - 1]:
                    continue

                g_l, h_l = g_cum[i - 1], h_cum[i - 1]
                g_r, h_r = G - g_l, H - h_l

                # min_child_weight constraint
                if h_l < self.min_child_weight or h_r < self.min_child_weight:
                    continue

                gain = 0.5 * (g_l**2 / (h_l + self.reg_lambda)
                              + g_r**2 / (h_r + self.reg_lambda)
                              - G**2   / (H   + self.reg_lambda)) - self.gamma

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = (sv[i - 1] + sv[i]) / 2.0

        return best_feat, best_thresh, best_gain

    # ── recursive build ──────────────────────────────
    def _build(self, X, g, h, depth):
        node = XGBNode()

        if depth >= self.max_depth or len(g) <= 1:
            node.is_leaf = True
            node.value   = self._leaf_weight(g, h)
            return node

        feat, thresh, gain = self._best_split(X, g, h)

        if feat is None or gain <= 0:
            node.is_leaf = True
            node.value   = self._leaf_weight(g, h)
            return node

        mask_l = X[:, feat] <= thresh
        mask_r = ~mask_l

        node.feature   = feat
        node.threshold = thresh
        node.left  = self._build(X[mask_l], g[mask_l], h[mask_l], depth + 1)
        node.right = self._build(X[mask_r], g[mask_r], h[mask_r], depth + 1)
        return node

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, depth=0)

    # ── predict one sample ───────────────────────────
    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# ─────────────────────────────────────────────────────────────────────────────

class XGBoostRegressorScratch:
    """
    XGBoost Regressor tự cài đặt.
    Loss: MSE  →  g_i = ŷ_i - y_i,  h_i = 1
    Hỗ trợ: subsample, colsample_bytree, learning_rate
    """

    def __init__(self,
                 n_estimators     = 300,
                 learning_rate    = 0.05,
                 max_depth        = 4,
                 min_child_weight = 5,
                 subsample        = 0.8,
                 colsample_bytree = 0.8,
                 reg_lambda       = 3.0,
                 gamma            = 0.3,
                 random_state     = 42):

        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_child_weight = min_child_weight
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda       = reg_lambda
        self.gamma            = gamma
        self.random_state     = random_state

        self.trees      = []        # list of (tree, col_indices)
        self.base_score = 0.5
        self.train_loss = []

    # ── gradient & hessian (MSE) ─────────────────────
    @staticmethod
    def _grad(y_pred, y_true):
        return y_pred - y_true          # dL/dŷ

    @staticmethod
    def _hess(y_pred, y_true):
        return np.ones_like(y_true)     # d²L/dŷ²

    # ── fit ─────────────────────────────────────────
    def fit(self, X, y, verbose=50):
        rng = np.random.RandomState(self.random_state)
        X   = np.array(X, dtype=np.float64)
        y   = np.array(y, dtype=np.float64)
        n, n_feat = X.shape

        self.base_score = y.mean()
        y_pred = np.full(n, self.base_score)

        for i in range(self.n_estimators):
            g = self._grad(y_pred, y)
            h = self._hess(y_pred, y)

            # row subsampling
            if self.subsample < 1.0:
                row_idx = rng.choice(n, int(n * self.subsample), replace=False)
            else:
                row_idx = np.arange(n)

            # column subsampling
            n_cols = max(1, int(n_feat * self.colsample_bytree))
            col_idx = rng.choice(n_feat, n_cols, replace=False)

            X_sub = X[np.ix_(row_idx, col_idx)]
            g_sub = g[row_idx]
            h_sub = h[row_idx]

            tree = XGBTree(
                max_depth        = self.max_depth,
                min_child_weight = self.min_child_weight,
                reg_lambda       = self.reg_lambda,
                gamma            = self.gamma,
            )
            tree.fit(X_sub, g_sub, h_sub)

            # update full prediction using ALL rows but selected cols
            update = tree.predict(X[:, col_idx])
            y_pred += self.learning_rate * update

            self.trees.append((tree, col_idx))

            # log
            loss = np.sqrt(np.mean((y_pred - y) ** 2))
            self.train_loss.append(loss)
            if verbose and (i + 1) % verbose == 0:
                print(f"  [{i+1:>4}/{self.n_estimators}] train RMSE = {loss:.4f}")

        return self

    # ── predict ─────────────────────────────────────
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        y_pred = np.full(X.shape[0], self.base_score)
        for tree, col_idx in self.trees:
            y_pred += self.learning_rate * tree.predict(X[:, col_idx])
        return y_pred

# =====================================================
# FEATURE ENGINEERING (giống bản gốc)
# =====================================================

# AQI lag
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

# AQI rolling (shift(1) trước để không rò rỉ)
for w in [6, 12, 24, 48]:
    df[f"aqi_roll_{w}"] = df["aqi"].shift(1).rolling(w).mean()

# AQI EMA
for span in [12, 24]:
    df[f"aqi_ema_{span}"] = df["aqi"].shift(1).ewm(span=span).mean()

# AQI trend
df["aqi_trend_1"]  = df["aqi_lag_1"] - df["aqi_lag_2"]
df["aqi_trend_6"]  = df["aqi_lag_1"] - df["aqi_lag_6"]
df["aqi_trend_24"] = df["aqi_lag_1"] - df["aqi_lag_24"]

# PM2.5 features
df["pm25_lag_1"]   = df["pm25"].shift(1)
df["pm25_lag_6"]   = df["pm25"].shift(6)
df["pm25_lag_24"]  = df["pm25"].shift(24)
df["pm25_roll_24"] = df["pm25"].shift(1).rolling(24).mean()
df["pm25_ema_24"]  = df["pm25"].shift(1).ewm(span=24).mean()

# PM10
df["pm10_lag_1"] = df["pm10"].shift(1)

# Weather lags
WEATHER_COLS = [
    "clouds", "precipitation", "pressure",
    "relative_humidity", "temperature", "uv_index", "wind_speed"
]
for col in WEATHER_COLS:
    df[f"{col}_lag_1"] = df[col].shift(1)

df = df.dropna().copy()
print(f"Sau khi tạo lag features: {len(df):,} hàng")

# =====================================================
# FEATURES & TARGET
# =====================================================

FEATURES = [
    # Weather
    "clouds_lag_1", "precipitation_lag_1", "pressure_lag_1",
    "relative_humidity_lag_1", "temperature_lag_1",
    "uv_index_lag_1", "wind_speed_lag_1",
    # Time
    "month", "hour", "day_of_week", "is_weekend", "is_rush_hour", "season",
    # AQI lag
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_lag_6", "aqi_lag_12",
    "aqi_lag_24", "aqi_lag_48", "aqi_lag_168",
    # AQI rolling
    "aqi_roll_6", "aqi_roll_12", "aqi_roll_24", "aqi_roll_48",
    # AQI EMA
    "aqi_ema_12", "aqi_ema_24",
    # AQI trend
    "aqi_trend_1", "aqi_trend_6", "aqi_trend_24",
    # PM2.5
    "pm25_lag_1", "pm25_lag_6", "pm25_lag_24",
    "pm25_roll_24", "pm25_ema_24",
    # PM10
    "pm10_lag_1",
]

TARGET = "aqi"

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

train = df[df["year"] < 2025].copy()
test  = df[df["year"] == 2025].copy()

X_train = train[FEATURES].values
y_train = train[TARGET].values

X_test  = test[FEATURES].values
y_test  = test[TARGET].values

print(f"\nTrain: {len(train):,} mẫu")
print(f"Test : {len(test):,} mẫu")
print(f"Số features: {len(FEATURES)}")

# =====================================================
# TRAIN XGBoost TỰ CÀI ĐẶT
# =====================================================

print("\n⏳ Training XGBoost (scratch)...")
model = XGBoostRegressorScratch(
    n_estimators     = 300,
    learning_rate    = 0.05,
    max_depth        = 4,
    min_child_weight = 5,
    subsample        = 0.70,
    colsample_bytree = 0.70,
    reg_lambda       = 5.0,
    gamma            = 0.5,
    random_state     = 42,
)
model.fit(X_train, y_train, verbose=50)
print("✅ Done")

# =====================================================
# ĐÁNH GIÁ
# =====================================================

train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:10s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return rmse, mae, r2

print("\n" + "=" * 50)
print("KẾT QUẢ XGBoost (scratch)")
print("=" * 50)
train_rmse, train_mae, train_r2 = metrics(y_train, train_pred, "Train")
test_rmse,  test_mae,  test_r2  = metrics(y_test,  test_pred,  "Test")
print(f"  Gap (Train R² – Test R²) = {train_r2 - test_r2:.4f}")

# =====================================================
# VISUALIZATION
# =====================================================

test_df = test.copy()
test_df["pred"] = test_pred

monthly = test_df.groupby("month").agg(
    actual=("aqi",  "mean"),
    pred  =("pred", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "XGBoost (tự cài đặt) – Dự báo AQI Hà Nội 2025",
    fontsize=14, fontweight="bold"
)

# Subplot 1 – Monthly predicted vs actual
ax = axes[0]
ax.plot(monthly["month"], monthly["actual"], "ko-", lw=2, label="Thực tế")
ax.plot(monthly["month"], monthly["pred"],   "rs--", lw=2, label="XGBoost (scratch)")
ax.set_xlabel("Tháng (2025)")
ax.set_ylabel("AQI trung bình")
ax.set_title("Predicted vs Actual theo tháng")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(monthly["month"])

# Subplot 2 – Scatter
ax = axes[1]
ax.scatter(y_test, test_pred, alpha=0.25, s=6, color="#1f77b4")
lim = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
ax.plot(lim, lim, "k--", lw=1.5, label="Perfect fit")
ax.set_xlabel("AQI thực tế")
ax.set_ylabel("AQI dự báo")
ax.set_title(f"Scatter Plot (R²={test_r2:.4f})")
ax.legend()
ax.grid(alpha=0.3)

# Subplot 3 – Training loss curve
ax = axes[2]
ax.plot(range(1, len(model.train_loss) + 1), model.train_loss, color="#2ca02c", lw=1.5)
ax.set_xlabel("Số cây (n_estimators)")
ax.set_ylabel("Train RMSE")
ax.set_title("Learning Curve")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
