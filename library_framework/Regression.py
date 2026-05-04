import pandas as pd
import numpy as np
import warnings
import os # BỔ SUNG: Để xử lý đường dẫn hệ thống
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

def find_data_file(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

file_name = 'hanoi_aqi_cleaned.csv'
file_path = find_data_file(file_name)

if file_path:
    print(f" Đã tìm thấy file tại: {file_path}")
    df = pd.read_csv(file_path)
else:
    # Nếu không tìm thấy tự động, yêu cầu nhập thủ công để tránh crash
    print(f"❌ Không tìm thấy file '{file_name}' tự động.")
    manual_path = input("Vui lòng dán đường dẫn đầy đủ của file .csv vào đây: ").strip('"')
    df = pd.read_csv(manual_path)

# Dọn dẹp tên cột và định dạng thời gian
df.columns = df.columns.str.strip().str.lower()
df['local_time'] = pd.to_datetime(df['local_time'])

# ══════════════════════════════════════════════════════
# 2. ĐỊNH NGHĨA BIẾN & HUẤN LUYỆN (Giữ nguyên)
# ══════════════════════════════════════════════════════
FEATURES = [
    'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
    'clouds', 'precipitation', 'pressure',
    'relative_humidity', 'temperature', 'uv_index', 'wind_speed',
    'month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season'
]
TARGET = 'aqi'

# Tách tập train (Sử dụng dữ liệu trước 2025 để học)
train = df[df['year'] < 2025].copy()
X_train = train[FEATURES]
y_train = train[TARGET]

rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
print("\n--- Hệ thống đang học dữ liệu khí hậu Hà Nội... ---")
rf_model.fit(X_train, y_train)

# ══════════════════════════════════════════════════════
# 3. QUY TẮC QUYẾT ĐỊNH (Giữ nguyên)
# ══════════════════════════════════════════════════════
AQI_RULES = {
    'Good': {'range': (0, 50), 'Trẻ em': '✅ Hoạt động bình thường', 'Người già': '✅ Hoạt động bình thường', 'Bệnh hô hấp': '✅ Hoạt động bình thường', 'Người khỏe mạnh': '✅ Hoạt động bình thường'},
    'Moderate': {'range': (51, 100), 'Trẻ em': '⚠️ Hạn chế vận động mạnh', 'Người già': '⚠️ Tránh vận động mạnh', 'Bệnh hô hấp': '⚠️ Theo dõi triệu chứng', 'Người khỏe mạnh': '✅ Hoạt động bình thường'},
    'Unhealthy for sensitive groups': {'range': (101, 150), 'Trẻ em': '🔶 Đeo khẩu trang', 'Người già': '🔶 Đeo khẩu trang N95', 'Bệnh hô hấp': '🔴 Tránh ra ngoài', 'Người khỏe mạnh': '⚠️ Hạn chế vận động mạnh'},
    'Unhealthy': {'range': (151, 200), 'Trẻ em': '🔴 Không nên ra ngoài', 'Người già': '🔴 Ở trong nhà', 'Bệnh hô hấp': '🔴 Ở trong nhà hoàn toàn', 'Người khỏe mạnh': '🔶 Đeo N95 nếu cần'},
    'Very Unhealthy': {'range': (201, 300), 'Trẻ em': '🚨 Ở trong nhà hoàn toàn', 'Người già': '🚨 Ở trong nhà hoàn toàn', 'Bệnh hô hấp': '🚨 Chuẩn bị thuốc cấp cứu', 'Người khỏe mạnh': '🔴 Tránh hoạt động ngoài trời'},
    'Hazardous': {'range': (301, 999), 'Trẻ em': '🚨 Cấm ra ngoài', 'Người già': '🚨 Liên hệ y tế ngay', 'Bệnh hô hấp': '🚨 Cần hỗ trợ khẩn cấp', 'Người khỏe mạnh': '🚨 Không ra ngoài'}
}

def get_recommendation(aqi_value):
    for cat, info in AQI_RULES.items():
        if info['range'][0] <= aqi_value <= info['range'][1]:
            return {'Mức độ': cat, **info}
    return None

# ══════════════════════════════════════════════════════
# 4. GIAO DIỆN DỰ BÁO (Hỗ trợ 2026)
# ══════════════════════════════════════════════════════
def start_aqi_app():
    print("\n" + "═"*60)
    print("   ỨNG DỤNG DỰ BÁO AQI HÀ NỘI - HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH")
    print("═"*60)
    
    try:
        date_input = input("\n📅 Nhập ngày (YYYY-MM-DD), VD: 2026-05-19: ")
        hour_input = int(input("⏰ Nhập giờ dự báo (0-23): "))
        
        target_date = pd.to_datetime(date_input)
        
        # Lấy giá trị trung bình khí tượng để làm dữ liệu nền
        weather_cols = [c for c in FEATURES if c not in ['month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season']]
        avg_weather = df[weather_cols].mean().to_dict()
        
        user_input = {
            **avg_weather,
            'month': target_date.month,
            'hour': hour_input,
            'day_of_week': target_date.dayofweek,
            'is_weekend': 1 if target_date.dayofweek >= 5 else 0,
            'is_rush_hour': 1 if hour_input in [7, 8, 9, 16, 17, 18, 19] else 0,
            'season': (target_date.month % 12 // 3) + 1
        }
        
        input_df = pd.DataFrame([user_input])[FEATURES]
        pred_aqi = rf_model.predict(input_df)[0]
        rec = get_recommendation(round(pred_aqi))
        
        print("\n" + "─"*60)
        print(f"📊 KẾT QUẢ DỰ BÁO: AQI ~ {round(pred_aqi, 1)} ({rec['Mức độ']})")
        print("─"*60)
        print(f"📢 KHUYẾN NGHỊ:")
        print(f"   • Trẻ em:          {rec['Trẻ em']}")
        print(f"   • Người già:       {rec['Người già']}")
        print(f"   • Nhóm bệnh hô hấp:{rec['Bệnh hô hấp']}")
        print(f"   • Người khỏe mạnh: {rec['Người khỏe mạnh']}")
        print("═"*60)

    except Exception as e:
        print(f"\n Lỗi: {e}")

if __name__ == "__main__":
    start_aqi_app()