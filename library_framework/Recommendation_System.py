import joblib
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

# ---------- LAGS ----------
df['aqi_lag_1']   = df['aqi'].shift(1)
df['aqi_lag_2']   = df['aqi'].shift(2)
df['aqi_lag_3']   = df['aqi'].shift(3)


df['aqi_lag_6']   = df['aqi'].shift(6)
df['aqi_lag_12']  = df['aqi'].shift(12)


df['aqi_lag_24']  = df['aqi'].shift(24)
df['aqi_lag_48']  = df['aqi'].shift(48)


df['aqi_lag_168'] = df['aqi'].shift(168)


# --- AQI ROLLING & EMA ---
df['aqi_roll_6']  = df['aqi'].shift(1).rolling(6).mean()
df['aqi_roll_12'] = df['aqi'].shift(1).rolling(12).mean()
df['aqi_roll_24'] = df['aqi'].shift(1).rolling(24).mean()
df['aqi_roll_48'] = df['aqi'].shift(1).rolling(48).mean()
df['aqi_ema_12']  = df['aqi'].shift(1).ewm(span=12).mean()
df['aqi_ema_24']  = df['aqi'].shift(1).ewm(span=24).mean()


# --- AQI TREND ---
df['aqi_trend_1']  = df['aqi_lag_1'] - df['aqi_lag_2']
df['aqi_trend_6']  = df['aqi_lag_1'] - df['aqi_lag_6']
df['aqi_trend_24'] = df['aqi_lag_1'] - df['aqi_lag_24']


#  PM2.5 FEATURES 
df['pm25_lag_1'] = df['pm25'].shift(1)
df['pm25_lag_6'] = df['pm25'].shift(6)
df['pm25_lag_24'] = df['pm25'].shift(24)

df['pm25_roll_24'] = df['pm25'].shift(1).rolling(24).mean()
df['pm25_ema_24']  = df['pm25'].shift(1).ewm(span=24).mean()

df['pm10_lag_1'] = df['pm10'].shift(1)

WEATHER_LAG_COLS = ['clouds', 'precipitation', 'pressure', 'relative_humidity', 'temperature', 'uv_index', 'wind_speed']
for col in WEATHER_LAG_COLS:
    df[f'{col}_lag_1'] = df[col].shift(1)

df = df.dropna().copy()

print(f" Sau khi tạo lag features: {len(df):,} hàng")

# =====================================================
# FEATURES & TARGET
# =====================================================

FEATURES = [
    # WEATHER
    'clouds_lag_1', 'precipitation_lag_1', 'pressure_lag_1', 
    'relative_humidity_lag_1', 'temperature_lag_1', 'uv_index_lag_1', 'wind_speed_lag_1',

    # TIME
    'month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season',

    # AQI LAGS
    'aqi_lag_1', 'aqi_lag_2', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_12', 'aqi_lag_24', 'aqi_lag_48', 'aqi_lag_168',

    # AQI ROLLING & EMA & TREND
    'aqi_roll_6', 'aqi_roll_12', 'aqi_roll_24', 'aqi_roll_48',
    'aqi_ema_12', 'aqi_ema_24',
    'aqi_trend_1', 'aqi_trend_6', 'aqi_trend_24',

    # PM2.5 & PM10 LAGS
    'pm25_lag_1', 'pm25_lag_6', 'pm25_lag_24', 'pm25_roll_24', 'pm25_ema_24',
    'pm10_lag_1'
]

TARGET = 'aqi'

train   = df[df['year'] < 2025].copy()
test    = df[df['year'] >= 2025].copy()
X_train = train[FEATURES]
y_train = train[TARGET]

print(f"\nTrain: {len(train):,} hàng (2022–2024)")

# Tải mô hình đã lưu từ trước
MODEL_PATH = "library_framework/best_model.pkl"
if os.path.exists(MODEL_PATH):
    print("⏳ Đang tải model đã train...")
    xgb_model = joblib.load(MODEL_PATH)
    print("✅ Model đã được load thành công!\n")
else:
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}!")

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
WEEKDAY_MAP = {
    0: 'Thứ Hai', 1: 'Thứ Ba', 2: 'Thứ Tư', 
    3: 'Thứ Năm', 4: 'Thứ Sáu', 5: 'Thứ Bảy', 6: 'Chủ Nhật'
}

cases = []

for season_code, season_name in SEASON_MAP.items():
    season_subset = test[test['season'] == season_code].copy()
    
    if len(season_subset) == 0:
        continue
        
    season_subset = season_subset.sort_values('aqi').reset_index(drop=True)
    
    # Chọn Mốc vị trí 25% (Đại diện ngày không khí sạch) và Mốc 85% (Đại diện ngày ô nhiễm đỉnh điểm của mùa)
    idx_selected = [int(len(season_subset) * 0.25), int(len(season_subset) * 0.85)]
    
    for idx in idx_selected:
        if idx >= len(season_subset): 
            continue
            
        row = season_subset.iloc[idx]
        
        month_val       = int(row['month'])
        hour_val        = int(row['hour'])
        day_of_week_val = int(row['day_of_week'])
        is_weekend_val  = int(row['is_weekend'])
        is_rush_val     = int(row['is_rush_hour'])
        weekday_name = WEEKDAY_MAP.get(day_of_week_val, f"Thứ {day_of_week_val}")

        matched_row = df[df['local_time'] == row['local_time']]
        
        if not matched_row.empty and not pd.isna(matched_row['aqi_lag_1'].values[0]):
            input_features_dict = matched_row[FEATURES].iloc[0].to_dict()
            source_type = f"Dữ liệu chuỗi trễ thực tế"
            
            temp_show = matched_row['temperature_lag_1'].values[0]
            rh_show   = matched_row['relative_humidity_lag_1'].values[0]
            ws_show   = matched_row['wind_speed_lag_1'].values[0]
            pm25_show = matched_row['pm25_lag_1'].values[0]
            pm10_show = matched_row['pm10_lag_1'].values[0]
        else:
            mask = (train['month'] == month_val) & (train['hour'] == hour_val)
            if mask.sum() == 0: 
                mask = train['month'] == month_val
                
            input_features_dict = train[mask][FEATURES].mean().to_dict()
            source_type = f"Ước lượng trung bình lịch sử (Tập Train)"
            
            temp_show = input_features_dict.get('temperature_lag_1', 25.0)
            rh_show   = input_features_dict.get('relative_humidity_lag_1', 75.0)
            ws_show   = input_features_dict.get('wind_speed_lag_1', 1.5)
            pm25_show = input_features_dict.get('pm25_lag_1', 40.0)
            pm10_show = input_features_dict.get('pm10_lag_1', 60.0)

        input_features_dict.update({
            'month':        month_val,
            'hour':         hour_val,
            'day_of_week':  day_of_week_val,
            'is_weekend':   is_weekend_val,
            'is_rush_hour': is_rush_val,
            'season':       season_code
        })
        
        input_df = pd.DataFrame([input_features_dict])[FEATURES]
        
        pred_aqi = round(float(xgb_model.predict(input_df)[0]), 1)
        
        pred_aqi_rounded = int(np.round(pred_aqi))
        level            = get_aqi_level(pred_aqi_rounded)
        
        time_ctx = get_time_context(hour_val, is_rush_val, is_weekend_val)
        ctx_advice = get_context_advice(pred_aqi_rounded, time_ctx, season_name, hour_val)

        cases.append({
            'Thời điểm':     str(row['local_time'])[:13] + 'h',
            'Thứ':           weekday_name,
            'AQI thực tế':   row['aqi'],
            'AQI dự báo':    pred_aqi,
            'Mức độ':        level,
            'Mùa':           season_name,
            'Thời điểm ctx': time_ctx,
            'Khuyến nghị':   AQI_RULES[level],
            'Context':       ctx_advice,
            'Nguồn':         source_type,
            'Nhiệt độ':      round(temp_show, 1),
            'Độ ẩm':         round(rh_show, 1),
            'Gió':           round(ws_show, 1),
            'PM2.5':         round(pm25_show, 1),
            'PM10':          round(pm10_show, 1)
        })

print("\n" + "="*75)
print("HỆ THỐNG KHUYẾN NGHỊ THÔNG MINH — KIỂM THỬ ĐẠI DIỆN ĐỦ 4 MÙA TẬP TEST 2025")
print("="*75)

for i, case in enumerate(cases, 1):
    rec = case['Khuyến nghị']
    print(f"\n{'─'*75}")
    print(f"Case {i}: [{case['Mùa']}] {case['Thời điểm']} ({case['Thứ']}) |  AQI thực tế={case['AQI thực tế']}  |  AQI dự báo={case['AQI dự báo']}")
    print(f"  Mức độ dự báo   : {case['Mức độ']}  |  Khung giờ ngữ cảnh: {case['Thời điểm ctx']}")
    print(f"  ℹ️  Nguồn bốc dữ liệu đầu vào: {case['Nguồn']}")
    print(f"  🌡️  [1h Trước] Nhiệt độ: {case['Nhiệt độ']:.1f}°C  |  Độ ẩm: {case['Độ ẩm']:.0f}%  |  Gió: {case['Gió']:.1f} m/s")
    print(f"  🏭  [1h Trước] PM2.5: {case['PM2.5']}  |  PM10: {case['PM10']}")
    print(f"\n  👥 KHUYẾN NGHỊ THEO NHÓM ĐỐI TƯỢNG:")
    print(f"     • Trẻ em           : {rec['Trẻ em']}")
    print(f"     • Người già        : {rec['Người già']}")
    print(f"     • Bệnh hô hấp      : {rec['Bệnh hô hấp']}")
    print(f"     • Người khỏe mạnh  : {rec['Người khỏe mạnh']}")
    print(f"\n  💡 KHUYẾN NGHỊ NGỮ CẢNH THÔNG MINH (CONTEXT-AWARE):")
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
