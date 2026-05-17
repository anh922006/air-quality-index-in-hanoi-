import pandas as pd
import numpy as np
import warnings
from sqlalchemy import create_engine 
import xgboost as xgb
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df = pd.read_sql('SELECT * FROM aqi_data', con=engine)
print(f"Đã đọc {len(df):,} hàng từ MySQL")

df.columns = df.columns.str.strip().str.lower()
df['local_time'] = pd.to_datetime(df['local_time'])

FEATURES = [
    'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    'clouds', 'precipitation', 'pressure',
    'relative_humidity', 'temperature', 'uv_index', 'wind_speed',
    'month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season'
]
TARGET = 'aqi'

train   = df[df['year'] < 2025].copy()
test    = df[df['year'] == 2025].copy()
X_train = train[FEATURES]
y_train = train[TARGET]

print(f"Train: {len(train):,} hàng (2022–2024)")
print(f"Test : {len(test):,} hàng (2025)")

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

print("⏳ Đang huấn luyện model...")
xgb_model.fit(X_train, y_train)
print("✅ Model sẵn sàng!")

# ══════════════════════════════════════════════════════
#  PHÂN LOẠI CONTEXT
# ══════════════════════════════════════════════════════
def get_time_context(hour, is_rush_hour, is_weekend):
    """Phân loại thời điểm trong ngày"""
    if is_rush_hour and not is_weekend:
        if 5 <= hour <= 10:
            return 'rush_morning'   # Cao điểm sáng
        else:
            return 'rush_evening'   # Cao điểm chiều
    elif 5 <= hour <= 11:
        return 'morning'            # Sáng thường
    elif 12 <= hour <= 17:
        return 'afternoon'          # Chiều
    elif 18 <= hour <= 22:
        return 'evening'            # Tối
    else:
        return 'night'              # Đêm khuya

def get_season_context(season):
    return {0: 'Đông', 1: 'Xuân', 2: 'Hạ', 3: 'Thu'}[season]

# ══════════════════════════════════════════════════════
#  KHUYẾN NGHỊ CƠ BẢN THEO AQI
# ══════════════════════════════════════════════════════
AQI_RULES = {
    'Good': {
        'range': (0, 50),
        'Trẻ em':          '✅ Hoạt động bình thường',
        'Người già':       '✅ Hoạt động bình thường',
        'Bệnh hô hấp':     '✅ Hoạt động bình thường',
        'Người khỏe mạnh': '✅ Hoạt động bình thường'
    },
    'Moderate': {
        'range': (51, 100),
        'Trẻ em':          '⚠️ Hạn chế vận động mạnh',
        'Người già':       '⚠️ Tránh vận động mạnh',
        'Bệnh hô hấp':     '⚠️ Theo dõi triệu chứng',
        'Người khỏe mạnh': '✅ Hoạt động bình thường'
    },
    'Unhealthy for sensitive groups': {
        'range': (101, 150),
        'Trẻ em':          '🔶 Đeo khẩu trang khi ra ngoài',
        'Người già':       '🔶 Đeo khẩu trang N95',
        'Bệnh hô hấp':     '🔴 Tránh ra ngoài',
        'Người khỏe mạnh': '⚠️ Hạn chế vận động mạnh ngoài trời'
    },
    'Unhealthy': {
        'range': (151, 200),
        'Trẻ em':          '🔴 Không nên ra ngoài',
        'Người già':       '🔴 Ở trong nhà',
        'Bệnh hô hấp':     '🔴 Ở trong nhà hoàn toàn',
        'Người khỏe mạnh': '🔶 Đeo N95 nếu bắt buộc ra ngoài'
    },
    'Very Unhealthy': {
        'range': (201, 300),
        'Trẻ em':          '🚨 Ở trong nhà hoàn toàn',
        'Người già':       '🚨 Ở trong nhà hoàn toàn',
        'Bệnh hô hấp':     '🚨 Chuẩn bị thuốc cấp cứu',
        'Người khỏe mạnh': '🔴 Tránh mọi hoạt động ngoài trời'
    },
    'Hazardous': {
        'range': (301, 999),
        'Trẻ em':          '🚨 Cấm ra ngoài',
        'Người già':       '🚨 Liên hệ y tế ngay',
        'Bệnh hô hấp':     '🚨 Cần hỗ trợ y tế khẩn cấp',
        'Người khỏe mạnh': '🚨 Không ra ngoài'
    }
}

def get_aqi_level(aqi_value):
    for cat, info in AQI_RULES.items():
        if info['range'][0] <= aqi_value <= info['range'][1]:
            return cat
    return 'Hazardous'

# ══════════════════════════════════════════════════════
#  KHUYẾN NGHỊ CONTEXT-AWARE (THÔNG MINH)
# ══════════════════════════════════════════════════════
def get_context_advice(aqi_value, time_ctx, season_name, hour):
    """
    Trả về lời khuyên thông minh dựa trên AQI + thời điểm + mùa
    """
    level = get_aqi_level(aqi_value)
    advice = []

    # ── Theo thời điểm ──────────────────────────────
    if time_ctx == 'rush_morning':
        if aqi_value > 100:
            advice.append("🚗 Giờ cao điểm sáng + ô nhiễm cao: tránh các tuyến Nguyễn Trãi, Tây Sơn, Giải Phóng — PM2.5 từ khói xe rất cao lúc này")
            advice.append("🚇 Nên dùng metro/xe buýt thay xe máy để giảm tiếp xúc trực tiếp")
        elif aqi_value > 50:
            advice.append("🚗 Giờ cao điểm sáng: đeo khẩu trang khi đi đường dù AQI ở mức trung bình")

    elif time_ctx == 'rush_evening':
        if aqi_value > 100:
            advice.append("🌆 Cao điểm chiều tối: AQI thường tệ nhất trong ngày lúc này — hạn chế tập thể dục ngoài trời")
            advice.append("🚗 Tránh đường Trần Duy Hưng, Láng Hạ, Đê La Thành giờ này")
        elif aqi_value > 50:
            advice.append("🌆 Cao điểm chiều: hạn chế ở ngoài đường lâu, di chuyển về nhà sớm")

    elif time_ctx == 'morning':
        if aqi_value <= 100:
            advice.append("🌅 Buổi sáng không cao điểm: thời điểm tốt để tập thể dục ngoài trời nếu AQI dưới 100")
        else:
            advice.append("🌅 Buổi sáng nhưng AQI cao: không nên chạy bộ hoặc tập ngoài trời")

    elif time_ctx == 'afternoon':
        if aqi_value <= 50:
            advice.append("☀️ Buổi chiều AQI tốt: có thể hoạt động ngoài trời bình thường")
        elif aqi_value <= 150:
            advice.append("☀️ Buổi chiều: UV Index thường cao nhất lúc 12–15h, kết hợp cả ô nhiễm — nên hạn chế ra ngoài")

    elif time_ctx == 'evening':
        if aqi_value > 150:
            advice.append("🌙 Buổi tối AQI cao: đóng cửa sổ, không mở cửa thông gió tự nhiên ban đêm")
        elif aqi_value <= 100:
            advice.append("🌙 Buổi tối AQI ổn: có thể mở cửa sổ thông gió nhẹ")

    elif time_ctx == 'night':
        if aqi_value > 100:
            advice.append("🌙 Ban đêm AQI cao: bật máy lọc không khí chế độ im lặng khi ngủ, đóng kín cửa sổ")
            advice.append("💤 Người có bệnh hô hấp: nên đặt máy lọc gần đầu giường")
        else:
            advice.append("🌙 Ban đêm AQI tốt: yên tâm ngủ, có thể mở hé cửa sổ thông gió nhẹ")

    # ── Theo mùa ────────────────────────────────────
    if season_name == 'Đông':
        if aqi_value > 100:
            advice.append("❄️ Mùa Đông: nghịch nhiệt khiến bụi mịn không thoát được lên cao — AQI thường xấu nhất năm, đặc biệt sáng sớm")
        if aqi_value > 150:
            advice.append("❄️ Mùa Đông AQI rất cao: hạn chế mở cửa sổ cả ngày, dùng máy lọc liên tục")

    elif season_name == 'Hạ':
        if aqi_value <= 100:
            advice.append("☀️ Mùa Hạ thường AQI tốt hơn: mưa nhiều giúp rửa bụi, nhưng O3 và UV Index cao — tránh ra ngoài 11h–15h")
        if aqi_value > 100:
            advice.append("☀️ Mùa Hạ AQI cao bất thường: có thể do đợt nắng hạn kéo dài hoặc cháy rừng")

    elif season_name == 'Xuân':
        advice.append("🌸 Mùa Xuân: độ ẩm cao, sương mù nhiều — bụi mịn dễ bị giữ lại trong không khí ẩm")

    elif season_name == 'Thu':
        if aqi_value > 100:
            advice.append("🍂 Mùa Thu: gió mùa Đông Bắc bắt đầu — ô nhiễm có xu hướng tăng dần cuối mùa Thu")

    # ── Lời khuyên chung theo mức AQI ───────────────
    if level == 'Good':
        advice.append("🍃 Không khí trong lành: tăng cường thông gió, mở cửa sổ thoải mái")
    elif level == 'Moderate':
        advice.append("🌤️ AQI trung bình: có thể mở cửa sổ nhưng hạn chế nếu nhà gần đường lớn")
    elif level == 'Unhealthy for sensitive groups':
        advice.append("🏠 Nên đóng bớt cửa sổ, dùng máy lọc không khí nếu có")
    elif level == 'Unhealthy':
        advice.append("🚫 Đóng chặt cửa sổ, bật máy lọc không khí liên tục")
    elif level == 'Very Unhealthy':
        advice.append("🚨 Đóng kín toàn bộ cửa sổ, tránh mọi khe hở thông gió")
    elif level == 'Hazardous':
        advice.append("🚨 Cực kỳ nguy hiểm — đóng kín nhà cửa, lọc khí tối đa, hạn chế ra ngoài tuyệt đối")

    return advice

# ══════════════════════════════════════════════════════
#  DEMO — 6 CASE MINH HỌA
# ══════════════════════════════════════════════════════
SEASON_MAP = {0: 'Đông', 1: 'Xuân', 2: 'Hạ', 3: 'Thu'}

cases = []
for cat in ['Good', 'Moderate', 'Unhealthy for sensitive groups',
            'Unhealthy', 'Very Unhealthy', 'Hazardous']:
    test['aqi_category'] = test['aqi_category'].str.strip()
    subset = test[test['aqi_category'] == cat]
    if len(subset) > 0:
        median_aqi = subset['aqi'].median()
        row = subset.iloc[(subset['aqi'] - median_aqi).abs().argsort().iloc[0]]
        pred_aqi = xgb_model.predict(pd.DataFrame([row[FEATURES]]))[0]
        level    = get_aqi_level(round(pred_aqi))
        time_ctx = get_time_context(row['hour'], row['is_rush_hour'], row['is_weekend'])
        season_n = SEASON_MAP.get(row['season'], 'Không rõ')
        ctx_advice = get_context_advice(round(pred_aqi), time_ctx, season_n, row['hour'])

        cases.append({
            'Thời điểm':   str(row['local_time'])[:13] + 'h',
            'AQI thực tế': row['aqi'],
            'AQI dự báo':  round(pred_aqi, 1),
            'Mức độ':      level,
            'Mùa':         season_n,
            'Thời điểm ctx': time_ctx,
            'Khuyến nghị': AQI_RULES[level],
            'Context':     ctx_advice
        })

print("\n" + "="*65)
print("HỆ THỐNG KHUYẾN NGHỊ THÔNG MINH — 6 CASE MINH HỌA")
print("="*65)

for i, case in enumerate(cases, 1):
    rec = case['Khuyến nghị']
    print(f"\n{'─'*65}")
    print(f"Case {i}: {case['Thời điểm']}  |  AQI thực tế={case['AQI thực tế']}  |  AQI dự báo={case['AQI dự báo']}")
    print(f"  Mức độ    : {case['Mức độ']}  |  Mùa: {case['Mùa']}  |  Khung giờ: {case['Thời điểm ctx']}")
    print(f"\n  👥 KHUYẾN NGHỊ THEO NHÓM:")
    print(f"     • Trẻ em           : {rec['Trẻ em']}")
    print(f"     • Người già        : {rec['Người già']}")
    print(f"     • Bệnh hô hấp      : {rec['Bệnh hô hấp']}")
    print(f"     • Người khỏe mạnh  : {rec['Người khỏe mạnh']}")
    print(f"\n  💡 KHUYẾN NGHỊ THÔNG MINH (CONTEXT-AWARE):")
    for tip in case['Context']:
        print(f"     → {tip}")

rec_table = []
for cat, info in AQI_RULES.items():
    lo, hi = info['range']
    rec_table.append({
        'Mức AQI':     f"{lo}–{'500+' if hi >= 999 else hi}",
        'Danh mục':    cat,
        'Trẻ em':      info['Trẻ em'],
        'Người già':   info['Người già'],
        'Bệnh hô hấp': info['Bệnh hô hấp'],
        'Khỏe mạnh':   info['Người khỏe mạnh'],
    })
pd.DataFrame(rec_table).to_csv('recommendation_table.csv', index=False, encoding='utf-8-sig')
