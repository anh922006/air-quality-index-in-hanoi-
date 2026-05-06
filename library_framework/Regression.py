import pandas as pd
import numpy as np
import warnings
import os
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 1. ĐỌC DỮ LIỆU
# ══════════════════════════════════════════════════════
def find_data_file(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

file_name = 'hanoi_aqi_cleaned.csv'
file_path = find_data_file(file_name)

if file_path:
    print(f"✅ Đã tìm thấy file tại: {file_path}")
    df = pd.read_csv(file_path)
else:
    print(f"❌ Không tìm thấy file '{file_name}' tự động.")
    manual_path = input("Vui lòng dán đường dẫn đầy đủ của file .csv: ").strip('"')
    df = pd.read_csv(manual_path)

df.columns = df.columns.str.strip().str.lower()
df['local_time'] = pd.to_datetime(df['local_time'])

# ══════════════════════════════════════════════════════
# 2. FEATURES & TRAIN MODEL
# ══════════════════════════════════════════════════════
FEATURES = [
    'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    'clouds', 'precipitation', 'pressure',
    'relative_humidity', 'temperature', 'uv_index', 'wind_speed',
    'month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season'
]
WEATHER_COLS = [c for c in FEATURES if c not in
                ['month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season']]
TARGET = 'aqi'

train = df[df['year'] < 2025].copy()
X_train = train[FEATURES]
y_train = train[TARGET]

rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
print("\n--- Hệ thống đang học dữ liệu khí hậu Hà Nội... ---")
rf_model.fit(X_train, y_train)
print("✅ Hoàn thành huấn luyện!")

# ══════════════════════════════════════════════════════
# 3. HÀM PHỤ TRỢ
# ══════════════════════════════════════════════════════
def get_season(month):
    """Đúng theo file clean: 0=Đông, 1=Xuân, 2=Hè, 3=Thu"""
    if month in [12, 1, 2]:  return 0  # Đông
    elif month in [3, 4, 5]: return 1  # Xuân
    elif month in [6, 7, 8]: return 2  # Hè
    else:                     return 3  # Thu

def get_is_rush_hour(hour):
    """Đúng theo file clean: sáng 6-9h, chiều 17-20h"""
    return 1 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0

# ══════════════════════════════════════════════════════
# 4. QUY TẮC KHUYẾN NGHỊ
# ══════════════════════════════════════════════════════
AQI_RULES = {
    'Good': {
        'range': (0, 50),
        'Hành động':        '🍃 Không khí trong lành. Tăng cường thông gió, mở cửa sổ.',
        'Trẻ em':           '✅ Hoạt động bình thường',
        'Người già':        '✅ Hoạt động bình thường',
        'Bệnh hô hấp':      '✅ Hoạt động bình thường',
        'Người khỏe mạnh':  '✅ Hoạt động bình thường'
    },
    'Moderate': {
        'range': (51, 100),
        'Hành động':        '🌤️ Có thể mở cửa sổ, hạn chế nếu nhà gần đường lớn.',
        'Trẻ em':           '⚠️ Hạn chế vận động mạnh',
        'Người già':        '⚠️ Tránh vận động mạnh',
        'Bệnh hô hấp':      '⚠️ Theo dõi triệu chứng',
        'Người khỏe mạnh':  '✅ Hoạt động bình thường'
    },
    'Unhealthy for sensitive groups': {
        'range': (101, 150),
        'Hành động':        '🏠 Đóng bớt cửa sổ. Dùng máy lọc không khí nếu có.',
        'Trẻ em':           '🔶 Đeo khẩu trang khi ra ngoài',
        'Người già':        '🔶 Đeo khẩu trang N95',
        'Bệnh hô hấp':      '🔴 Tránh ra ngoài',
        'Người khỏe mạnh':  '⚠️ Hạn chế vận động mạnh ngoài trời'
    },
    'Unhealthy': {
        'range': (151, 200),
        'Hành động':        '🚫 Đóng chặt cửa sổ. Bật máy lọc không khí.',
        'Trẻ em':           '🔴 Không nên ra ngoài',
        'Người già':        '🔴 Ở trong nhà',
        'Bệnh hô hấp':      '🔴 Ở trong nhà hoàn toàn',
        'Người khỏe mạnh':  '🔶 Đeo N95 nếu bắt buộc ra ngoài'
    },
    'Very Unhealthy': {
        'range': (201, 300),
        'Hành động':        '🚨 Nguy hại! Đóng kín toàn bộ cửa sổ.',
        'Trẻ em':           '🚨 Ở trong nhà hoàn toàn',
        'Người già':        '🚨 Ở trong nhà hoàn toàn',
        'Bệnh hô hấp':      '🚨 Chuẩn bị thuốc cấp cứu',
        'Người khỏe mạnh':  '🔴 Tránh mọi hoạt động ngoài trời'
    },
    'Hazardous': {
        'range': (301, 999),
        'Hành động':        '🚨 Cực kỳ nguy hiểm! Đóng kín nhà cửa, lọc khí tối đa.',
        'Trẻ em':           '🚨 Cấm ra ngoài',
        'Người già':        '🚨 Liên hệ y tế ngay',
        'Bệnh hô hấp':      '🚨 Cần hỗ trợ y tế khẩn cấp',
        'Người khỏe mạnh':  '🚨 Không ra ngoài'
    }
}

def get_recommendation(aqi_value):
    for cat, info in AQI_RULES.items():
        lo, hi = info['range']
        if lo <= aqi_value <= hi:
            return {'Mức độ': cat, **info}
    return None

# ══════════════════════════════════════════════════════
# 5. ỨNG DỤNG DỰ BÁO
# ══════════════════════════════════════════════════════
def start_aqi_app():
    print("\n" + "═"*60)
    print("   ỨNG DỤNG DỰ BÁO AQI HÀ NỘI - HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH")
    print("═"*60)

    try:
        date_input = input("\n📅 Nhập ngày (YYYY-MM-DD), VD: 2026-05-19: ").strip()
        hour_input = int(input("⏰ Nhập giờ dự báo (0-23): ").strip())

        if not (0 <= hour_input <= 23):
            print("❌ Giờ không hợp lệ, vui lòng nhập 0–23.")
            return

        target_date = pd.to_datetime(date_input)
        month  = target_date.month
        season = get_season(month)

        # Lấy trung bình thời tiết theo cùng tháng + giờ — chính xác hơn mean toàn bộ
        mask = (df['month'] == month) & (df['hour'] == hour_input)
        if mask.sum() == 0:
            # Fallback: chỉ theo tháng nếu không có đủ data
            mask = df['month'] == month
        avg_weather = df[mask][WEATHER_COLS].mean().to_dict()

        user_input = {
            **avg_weather,
            'month':        month,
            'hour':         hour_input,
            'day_of_week':  target_date.dayofweek,
            'is_weekend':   1 if target_date.dayofweek >= 5 else 0,
            'is_rush_hour': get_is_rush_hour(hour_input),
            'season':       season,
        }

        input_df = pd.DataFrame([user_input])[FEATURES]
        pred_aqi = rf_model.predict(input_df)[0]
        rec      = get_recommendation(round(pred_aqi))

        SEASON_NAME = {0: 'Đông', 1: 'Xuân', 2: 'Hè', 3: 'Thu'}
        rush_label  = "Giờ cao điểm" if user_input['is_rush_hour'] else "Không cao điểm"
        weekend_label = "Cuối tuần" if user_input['is_weekend'] else "Ngày thường"

        print("\n" + "─"*60)
        print(f"📅 Thời điểm  : {date_input} {hour_input:02d}:00  |  {SEASON_NAME[season]}  |  {weekend_label}  |  {rush_label}")
        print(f"🌡️  Nhiệt độ   : {avg_weather['temperature']:.1f}°C   |  Độ ẩm: {avg_weather['relative_humidity']:.0f}%   |  Gió: {avg_weather['wind_speed']:.1f} m/s")
        print(f"🏭 PM2.5      : {avg_weather['pm25']:.1f}   |  PM10: {avg_weather['pm10']:.1f}   |  NO2: {avg_weather['no2']:.1f}")
        print("─"*60)
        print(f"📊 AQI DỰ BÁO : {round(pred_aqi, 1)}  →  {rec['Mức độ']}")
        print(f"📢 Hành động  : {rec['Hành động']}")
        print("─"*60)
        print("👥 KHUYẾN NGHỊ THEO NHÓM:")
        print(f"   • Trẻ em           : {rec['Trẻ em']}")
        print(f"   • Người già        : {rec['Người già']}")
        print(f"   • Bệnh hô hấp      : {rec['Bệnh hô hấp']}")
        print(f"   • Người khỏe mạnh  : {rec['Người khỏe mạnh']}")
        print("═"*60)

    except ValueError:
        print("❌ Ngày hoặc giờ không hợp lệ. Vui lòng nhập đúng định dạng.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    start_aqi_app()