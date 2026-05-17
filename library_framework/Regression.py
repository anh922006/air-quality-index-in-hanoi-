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
print(f"✅ Đã đọc {len(df):,} hàng từ MySQL")

df.columns = df.columns.str.strip().str.lower()
df['local_time'] = pd.to_datetime(df['local_time'])
df = df.sort_values('local_time').reset_index(drop=True)

# ══════════════════════════════════════════════════════
# 2. FEATURES & TARGET
# ══════════════════════════════════════════════════════
# Dùng ô nhiễm real-time — đúng với thực tế có sensor
FEATURES = [
    'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    'clouds', 'precipitation', 'pressure',
    'relative_humidity', 'temperature', 'uv_index', 'wind_speed',
    'month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season'
]

WEATHER_COLS = [
    'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    'clouds', 'precipitation', 'pressure',
    'relative_humidity', 'temperature', 'uv_index', 'wind_speed'
]

TARGET = 'aqi'

train   = df[df['year'] < 2025].copy()
X_train = train[FEATURES]
y_train = train[TARGET]

print(f"\nTrain: {len(train):,} hàng (2022–2024)")

# ══════════════════════════════════════════════════════
# 3. TRAIN XGBOOST
# ══════════════════════════════════════════════════════
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
print("✅ Model sẵn sàng!\n")

# ══════════════════════════════════════════════════════
# 4. HÀM PHỤ TRỢ
# ══════════════════════════════════════════════════════
def get_season(month):
    if month in [12, 1, 2]:  return 0   # Đông
    elif month in [3, 4, 5]: return 1   # Xuân
    elif month in [6, 7, 8]: return 2   # Hè
    else:                     return 3   # Thu

def get_is_rush_hour(hour):
    return 1 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0

def get_time_context(hour, is_rush_hour, is_weekend):
    if is_rush_hour and not is_weekend:
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

SEASON_MAP   = {0: 'Đông', 1: 'Xuân', 2: 'Hè', 3: 'Thu'}
SEASON_EMOJI = {0: '❄️',   1: '🌸',   2: '☀️', 3: '🍂'}

# ══════════════════════════════════════════════════════
# 5. KHUYẾN NGHỊ THEO NHÓM NGƯỜI DÙNG
# ══════════════════════════════════════════════════════
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
        'Trẻ em':          '⚠️ Hạn chế vận động mạnh',
        'Người già':       '⚠️ Tránh vận động mạnh',
        'Bệnh hô hấp':     '⚠️ Theo dõi triệu chứng',
        'Người khỏe mạnh': '✅ Hoạt động bình thường'
    },
    'Unhealthy for sensitive groups': {
        'range': (101, 150), 'emoji': '🟠',
        'Trẻ em':          '🔶 Đeo khẩu trang khi ra ngoài',
        'Người già':       '🔶 Đeo khẩu trang N95',
        'Bệnh hô hấp':     '🔴 Tránh ra ngoài',
        'Người khỏe mạnh': '⚠️ Hạn chế vận động mạnh ngoài trời'
    },
    'Unhealthy': {
        'range': (151, 200), 'emoji': '🔴',
        'Trẻ em':          '🔴 Không nên ra ngoài',
        'Người già':       '🔴 Ở trong nhà',
        'Bệnh hô hấp':     '🔴 Ở trong nhà hoàn toàn',
        'Người khỏe mạnh': '🔶 Đeo N95 nếu bắt buộc ra ngoài'
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
        'Trẻ em':          '🚨 Cấm ra ngoài',
        'Người già':       '🚨 Liên hệ y tế ngay',
        'Bệnh hô hấp':     '🚨 Cần hỗ trợ y tế khẩn cấp',
        'Người khỏe mạnh': '🚨 Không ra ngoài'
    }
}

# ══════════════════════════════════════════════════════
# 6. KHUYẾN NGHỊ CONTEXT-AWARE
# ══════════════════════════════════════════════════════
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
    elif season_name == 'Hè':
        advice.append("☀️ Mùa Hè: O3 và UV Index cao — tránh ra ngoài lúc 11h–15h dù AQI có vẻ ổn")
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

# ══════════════════════════════════════════════════════
# 7. ỨNG DỤNG NHẬP NGÀY THÁNG DỰ BÁO
# ══════════════════════════════════════════════════════
def predict_by_datetime():
    print("\n" + "═"*62)
    print("  ỨNG DỤNG DỰ BÁO AQI HÀ NỘI — KHUYẾN NGHỊ THÔNG MINH")
    print("═"*62)

    try:
        date_input = input("\n📅 Nhập ngày (YYYY-MM-DD): ").strip()
        hour_input = int(input("⏰ Nhập giờ (0-23): ").strip())

        if not (0 <= hour_input <= 23):
            print("❌ Giờ không hợp lệ, vui lòng nhập từ 0 đến 23.")
            return

        target_date = pd.to_datetime(date_input)
        month       = target_date.month
        year        = target_date.year
        season      = get_season(month)
        is_weekend  = 1 if target_date.dayofweek >= 5 else 0
        is_rush     = get_is_rush_hour(hour_input)
        time_ctx    = get_time_context(hour_input, is_rush, is_weekend)
        season_name = SEASON_MAP[season]

        actual_data = df[(df['local_time'].dt.date == target_date.date()) & (df['hour'] == hour_input)]
        
        if year == 2025 and not actual_data.empty:
            print(f"📡 Đang sử dụng chỉ số thực tế (Real-time) từ hệ thống năm 2025...")
            # Lấy tất cả trừ cột AQI (Target)
            input_values = actual_data[WEATHER_COLS].iloc[0].to_dict()
            source_type = "Số liệu thực tế từ trạm đo"
        else:
            print(f"📚 Đang ước tính chỉ số dựa trên hồ sơ lịch sử (tháng {month}, {hour_input}h)...")
            # Lấy trung bình lịch sử từ tập Train (trước 2025)
            historical_data = df[df['year'] < 2025]
            mask = (historical_data['month'] == month) & (historical_data['hour'] == hour_input)
            if mask.sum() == 0:
                mask = historical_data['month'] == month
            input_values = historical_data[mask][WEATHER_COLS].mean().to_dict()
            source_type = "Số liệu trung bình lịch sử"

        user_input = {
            **input_values,
            'month':        month,
            'hour':         hour_input,
            'day_of_week':  target_date.dayofweek,
            'is_weekend':   is_weekend,
            'is_rush_hour': is_rush,
            'season':       season,
        }

        input_df         = pd.DataFrame([user_input])[FEATURES]
        pred_aqi         = xgb_model.predict(input_df)[0]
        pred_aqi_rounded = round(pred_aqi)
        level            = get_aqi_level(pred_aqi_rounded)
        rec              = AQI_RULES[level]
        ctx_tips         = get_context_advice(pred_aqi_rounded, time_ctx, season_name, hour_input)

        rush_label    = "Giờ cao điểm 🚦" if is_rush else "Không cao điểm"
        weekend_label = "Cuối tuần 🎉" if is_weekend else "Ngày thường"
        dow_names     = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']
        dow_label     = dow_names[target_date.dayofweek]

        print("\n" + "─"*62)
        print(f"📅  {date_input}  {hour_input:02d}:00  |  {dow_label}  |  {weekend_label}")
        print(f"🗓️   Mùa: {SEASON_EMOJI[season]} {season_name}  |  {rush_label}")
        print(f"🌡️   Nhiệt độ: {input_values['temperature']:.1f}°C  |  Độ ẩm: {input_values['relative_humidity']:.0f}%  |  Gió: {input_values['wind_speed']:.1f} m/s")
        print(f"🏭  PM2.5: {input_values['pm25']:.1f}  |  PM10: {input_values['pm10']:.1f}  |  NO2: {input_values['no2']:.1f}")
        print("─"*62)
        print(f"\n{rec['emoji']}  AQI DỰ BÁO: {round(pred_aqi, 1)}  →  {level}")
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

# ══════════════════════════════════════════════════════
# 8. CHẠY ỨNG DỤNG
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    predict_by_datetime()