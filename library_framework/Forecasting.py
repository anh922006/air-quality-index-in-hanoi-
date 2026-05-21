import pandas as pd
import numpy as np
import warnings
from sqlalchemy import create_engine
import xgboost as xgb
import os
from dotenv import load_dotenv
import joblib

warnings.filterwarnings('ignore')

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df = pd.read_sql('SELECT * FROM aqi_data', con=engine)
print(f"✅ Đã đọc {len(df):,} hàng từ MySQL")

df.columns = df.columns.str.strip().str.lower()
df['local_time'] = pd.to_datetime(df['local_time'])
df = df.sort_values('local_time').reset_index(drop=True)

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
X_train = train[FEATURES]
y_train = train[TARGET]

print(f"\nTrain: {len(train):,} hàng (2022–2024)")

MODEL_PATH = "library_framework/best_model.pkl"
if os.path.exists(MODEL_PATH):
    print("⏳ Đang tải model đã train...")
    xgb_model = joblib.load(MODEL_PATH)
    print("✅ Model đã được load thành công!\n")
else:
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}!")

# HÀM PHỤ TRỢ
def get_season(month):
    if month in [12, 1, 2]:  return 0   # Đông
    elif month in [3, 4, 5]: return 1   # Xuân
    elif month in [6, 7, 8]: return 2   # Hạ
    else:                     return 3   # Thu

def get_is_rush_hour(hour, is_weekend):
    if is_weekend:
        return 1 if hour in [17, 18, 19, 20] else 0
    return 1 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0

def get_time_context(hour, is_rush_hour, is_weekend):
    if is_rush_hour:
        return 'rush_morning' if 5 <= hour <= 10 else 'rush_evening'
    elif 5 <= hour <= 11:  return 'morning'
    elif 12 <= hour <= 17: return 'afternoon'
    elif 18 <= hour <= 22: return 'evening'
    else:                   return 'night'

def get_aqi_level(aqi_value):
    if aqi_value <= 50:    return 'Good'
    elif aqi_value <= 100: return 'Moderate'
    elif aqi_value <= 150: return 'Unhealthy for sensitive groups'
    elif aqi_value <= 200: return 'Unhealthy'
    elif aqi_value <= 300: return 'Very Unhealthy'
    else:                  return 'Hazardous'

SEASON_MAP   = {0: 'Đông', 1: 'Xuân', 2: 'Hạ', 3: 'Thu'}
SEASON_EMOJI = {0: '❄️',   1: '🌸',   2: '☀️', 3: '🍂'}

#  KHUYẾN NGHỊ THEO NHÓM NGƯỜI DÙNG
AQI_RULES = {
    'Good': {
        'range': (0, 50), 'emoji': '🟢',
        'Trẻ em':          '✅ Hoạt động bình thường',
        'Người già':       '✅ Hoạt động bình thường',
        'Bệnh hô hấp':     '✅ Hoạt động bình thường',
        'Người khỏe mạnh': '✅ Hoạt động bình thường'
    },
    'Moderate': {
        'range': (51, 100), 'emoji': '🟡',
        'Trẻ em':          '✅ Hoạt động bình thường',
        'Người già':       '✅ Hoạt động bình thường',
        'Bệnh hô hấp':     '🌤️ Theo dõi sức khỏe',
        'Người khỏe mạnh': '✅ Hoạt động bình thường'
    },
    'Unhealthy for sensitive groups': {
        'range': (101, 150), 'emoji': '🟠',
        'Trẻ em':          '🔶 Hạn chế hoạt động ngoài trời lâu',
        'Người già':       '🔶 Đeo khẩu trang khi ra ngoài',
        'Bệnh hô hấp':     '🔴 Hạn chế ra ngoài',
        'Người khỏe mạnh': '🌤️ Đeo khẩu trang khi đi đường'
    },
    'Unhealthy': {
        'range': (151, 200), 'emoji': '🔴',
        'Trẻ em':          '🔴 Không nên ra ngoài',
        'Người già':       '🔴 Ở trong nhà',
        'Bệnh hô hấp':     '🚨 Ở trong nhà hoàn toàn',
        'Người khỏe mạnh': '🔶 Đeo khẩu trang N95 khi ra ngoài'
    },
    'Very Unhealthy': {
        'range': (201, 300), 'emoji': '🟣',
        'Trẻ em':          '🚨 Ở trong nhà hoàn toàn',
        'Người già':       '🚨 Ở trong nhà hoàn toàn',
        'Bệnh hô hấp':     '🚨 Chuẩn bị thuốc cấp cứu',
        'Người khỏe mạnh': '🔴 Tránh mọi hoạt động ngoài trời'
    },
    'Hazardous': {
        'range': (301, 999), 'emoji': '⚫',
        'Trẻ em':          '🚫 Nghiêm cấm ra ngoài',
        'Người già':       '🚫 Liên hệ y tế nếu mệt mỏi',
        'Bệnh hô hấp':     '🚫 Cần hỗ trợ y tế khẩn cấp',
        'Người khỏe mạnh': '🚨 Ở trong nhà, đóng kín cửa sổ'
    }
}

#  KHUYẾN NGHỊ CONTEXT-AWARE
def get_context_advice(aqi_value, time_ctx, season_name, hour):
    level  = get_aqi_level(aqi_value)
    advice = []

    if time_ctx == 'rush_morning':
        if aqi_value > 100:
            advice.append("🚗 Giờ cao điểm sáng + ô nhiễm cao: tránh Nguyễn Trãi, Tây Sơn, Giải Phóng — PM2.5 từ khói xe rất cao")
            advice.append("🚇 Nên dùng metro/xe buýt thay xe máy để giảm tiếp xúc trực tiếp")
        else:
            advice.append("🚗 Giờ cao điểm sáng: đeo khẩu trang khi đi đường dù AQI ở mức trung bình")

    elif time_ctx == 'rush_evening':
        if aqi_value > 100:
            advice.append("🌆 Cao điểm chiều tối: AQI thường tệ nhất trong ngày — hạn chế tập thể dục ngoài trời")
            advice.append("🚗 Tránh đường Trần Duy Hưng, Láng Hạ, Đê La Thành giờ này")
        else:
            advice.append("🌆 Cao điểm chiều: hạn chế ở ngoài đường lâu, di chuyển về nhà sớm")

    elif time_ctx == 'morning':
        if aqi_value <= 100:
            advice.append("🌅 Buổi sáng không cao điểm: thời điểm tốt để tập thể dục ngoài trời")
        else:
            advice.append("🌅 Buổi sáng nhưng AQI cao: không nên chạy bộ hoặc tập ngoài trời")

    elif time_ctx == 'afternoon':
        if aqi_value <= 50:
            advice.append("☀️ Buổi chiều AQI tốt: có thể hoạt động ngoài trời bình thường")
        else:
            advice.append("☀️ Buổi chiều: UV Index cao nhất lúc 12–15h kết hợp ô nhiễm — nên hạn chế ra ngoài")

    elif time_ctx == 'evening':
        if aqi_value > 150:
            advice.append("🌙 Buổi tối AQI cao: đóng cửa sổ, không mở cửa thông gió tự nhiên")
        else:
            advice.append("🌙 Buổi tối AQI ổn: có thể mở cửa sổ thông gió nhẹ")

    elif time_ctx == 'night':
        if aqi_value > 100:
            advice.append("🌙 Ban đêm AQI cao: bật máy lọc không khí chế độ im lặng khi ngủ, đóng kín cửa sổ")
            advice.append("💤 Người có bệnh hô hấp: nên đặt máy lọc gần đầu giường")
        else:
            advice.append("🌙 Ban đêm AQI tốt: yên tâm ngủ, có thể mở hé cửa sổ thông gió nhẹ")

    if season_name == 'Đông' and aqi_value > 100:
        advice.append("❄️ Mùa Đông: nghịch nhiệt khiến bụi mịn tích tụ — AQI thường xấu nhất năm, đặc biệt sáng sớm")
    elif season_name == 'Hạ':
        advice.append("☀️ Mùa Hạ: O3 và UV Index cao — tránh ra ngoài lúc 11h–15h dù AQI có vẻ ổn")
    elif season_name == 'Xuân':
        advice.append("🌸 Mùa Xuân: độ ẩm cao, sương mù nhiều — bụi mịn dễ tích tụ trong không khí ẩm")
    elif season_name == 'Thu' and aqi_value > 100:
        advice.append("🍂 Mùa Thu: gió mùa Đông Bắc bắt đầu — ô nhiễm có xu hướng tăng cuối mùa")

    general = {
        'Good':                           "🍃 Không khí trong lành: tăng cường thông gió, mở cửa sổ thoải mái",
        'Moderate':                       "🌤️ AQI trung bình: có thể mở cửa sổ, hạn chế nếu nhà gần đường lớn",
        'Unhealthy for sensitive groups': "🏠 Nên đóng bớt cửa sổ, dùng máy lọc không khí nếu có",
        'Unhealthy':                      "🚫 Đóng chặt cửa sổ, bật máy lọc không khí liên tục",
        'Very Unhealthy':                 "🚨 Đóng kín toàn bộ cửa sổ, tránh mọi khe hở thông gió",
        'Hazardous':                      "🚨 Cực kỳ nguy hiểm — đóng kín nhà cửa, lọc khí tối đa"
    }
    advice.append(general[level])
    return advice

#  ỨNG DỤNG NHẬP NGÀY THÁNG DỰ BÁO
def predict_by_datetime():
    print("\n" + "═"*62)
    print("  ỨNG DỤNG DỰ BÁO AQI HÀ NỘI — KHUYẾN NGHỊ THÔNG MINH")
    print("═"*62)

    try:
        date_input = input("\n📅 Nhập ngày muốn xem dữ liệu (YYYY-MM-DD): ").strip()
        hour_input = int(input("⏰ Nhập giờ muốn xem dữ liệu (0-23): ").strip())

        if not (0 <= hour_input <= 23):
            print("❌ Giờ không hợp lệ, vui lòng nhập từ 0 đến 23.")
            return

        target_time = pd.to_datetime(f"{date_input} {hour_input:02d}:00:00")
        
        month       = target_time.month
        season      = get_season(month)
        is_weekend  = 1 if target_time.dayofweek >= 5 else 0
        is_rush     = get_is_rush_hour(hour_input, is_weekend)
        time_ctx    = get_time_context(hour_input, is_rush, is_weekend)
        season_name = SEASON_MAP[season]

        matched_row = df[df['local_time'] == target_time]

        if not matched_row.empty:
            print(f"📡 Tìm thấy lịch sử chuỗi thời gian liên tục tại mốc {target_time}...")
            user_input_features = matched_row[FEATURES].iloc[0].to_dict()
            
            # Trích xuất giá trị thực tế của 1 giờ trước để hiển thị 
            temp_show = matched_row['temperature_lag_1'].values[0]
            rh_show   = matched_row['relative_humidity_lag_1'].values[0]
            ws_show   = matched_row['wind_speed_lag_1'].values[0]
            pm25_show = matched_row['pm25_lag_1'].values[0]
            pm10_show = matched_row['pm10_lag_1'].values[0]
            source_type = "Số liệu lưu trữ chuỗi thời gian"
        else:
            # Phương án dự phòng nếu trúng ngày chưa có dữ liệu/ngày tương lai
            print(f"📚 Không tìm thấy chuỗi trễ liên tục. Đang tạo dữ liệu giả lập từ trung bình lịch sử tháng {month}...")
            mask = (df['month'] == month) & (df['hour'] == hour_input)
            if mask.sum() == 0:
                mask = df['month'] == month
            
            mean_values = df[mask][FEATURES].mean().to_dict()
            user_input_features = mean_values
            
            temp_show = mean_values.get('temperature_lag_1', 25.0)
            rh_show   = mean_values.get('relative_humidity_lag_1', 75.0)
            ws_show   = mean_values.get('wind_speed_lag_1', 1.5)
            pm25_show = mean_values.get('pm25_lag_1', 40.0)
            pm10_show = mean_values.get('pm10_lag_1', 60.0)
            source_type = "Số liệu ước lượng trung bình lịch sử"

        user_input_features.update({
            'month':        month,
            'hour':         hour_input,
            'day_of_week':  target_time.dayofweek,
            'is_weekend':   is_weekend,
            'is_rush_hour': is_rush,
            'season':       season
        })

        input_df         = pd.DataFrame([user_input_features])[FEATURES]
        pred_aqi         = round(float(xgb_model.predict(input_df)[0]), 1)
        pred_aqi_rounded = round(pred_aqi, 1)
        
        level            = get_aqi_level(pred_aqi_rounded)
        rec              = AQI_RULES[level]
        ctx_tips         = get_context_advice(pred_aqi_rounded, time_ctx, season_name, hour_input)

        rush_label    = "Giờ cao điểm 🚦" if is_rush else "Không cao điểm"
        weekend_label = "Cuối tuần 🎉" if is_weekend else "Ngày thường"
        dow_names     = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']
        dow_label     = dow_names[target_time.dayofweek]

        print("\n" + "─"*62)
        print(f"📅  {date_input}  {hour_input:02d}:00  |  {dow_label}  |  {weekend_label}")
        print(f"🗓️   Mùa: {SEASON_EMOJI[season]} {season_name}  |  {rush_label}")
        print(f"ℹ️   Nguồn gốc dữ liệu: {source_type}")
        print(f"🌡️   [1h Trước] Nhiệt độ: {temp_show:.1f}°C  |  Độ ẩm: {rh_show:.0f}%  |  Gió: {ws_show:.1f} m/s")
        print(f"🏭  [1h Trước] PM2.5: {pm25_show:.1f}  |  PM10: {pm10_show:.1f}")
        print("─"*62)
        print(f"\n{rec['emoji']}  AQI DỰ BÁO: {pred_aqi_rounded}  →  {level}")
        print("\n👥 KHUYẾN NGHỊ THEO NHÓM:")
        print(f"   • Trẻ em           : {rec['Trẻ em']}")
        print(f"   • Người già        : {rec['Người già']}")
        print(f"   • Bệnh hô hấp      : {rec['Bệnh hô hấp']}")
        print(f"   • Người khỏe mạnh  : {rec['Người khỏe mạnh']}")
        print("\n💡 KHUYẾN NGHỊ THÔNG MINH (CONTEXT-AWARE):")
        for tip in ctx_tips:
            print(f"   → {tip}")
        print("\n" + "═"*62)

        again = input("\n🔄 Dự báo thêm giờ khác? (y/n): ").strip().lower()
        if again == 'y':
            predict_by_datetime()

    except ValueError:
        print("❌ Ngày hoặc giờ không hợp lệ. Vui lòng nhập đúng định dạng YYYY-MM-DD và giờ từ 0–23.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    predict_by_datetime()