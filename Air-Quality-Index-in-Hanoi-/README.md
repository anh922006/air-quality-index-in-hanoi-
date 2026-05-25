# Air-Quality-Index-in-Hanoi-
Dựa vào data chỉ số chất lượng không khí từ 2022-2025 để xây dựng một Hệ thống Dự báo Chất lượng Không khí
1. Dự án (Project)
- Tên dự án: Hệ thống Dự báo Chất lượng Không khí
- Dự án phân tích và dự báo chỉ số chất lượng không khí tại Hà Nội sử dụng các thuật toán Học máy (Regression & Classification). Mục tiêu là đưa ra các cảnh báo sớm về ô nhiễm dựa trên dữ liệu khí tượng và nồng độ các chất khí.
- Dữ liệu: Hơn 30,000 bản ghi dữ liệu lịch sử từ năm 2022 đến đầu năm 2025 tại khu vực Hà Nội.
- Thành viên nhóm: Nguyễn Thị Phương Anh - B24DCTC005
                   Bùi Quang Hùng - B24DCTC039
                   Lưu Linh Linh - B24DCTC059
                   Tạ Minh Trường - B24DCTC113

2. Cấu trúc thư mục (Project Structure)
Air-Quality-Index-in-Hanoi/
├── data_raw/                     # Thư mục chứa 4 file CSV dữ liệu thô gốc (2022–2025)
├── clean/                        
│   ├── clean.ipynb               # Làm sạch dữ liệu
│   ├── hanoi_aqi_cleaned.csv     # Tệp dữ liệu đầu ra đã làm sạch (Cleaned Dataset)
│   └── sql.py                    # Mã nguồn thiết lập kết nối và nạp dữ liệu sạch vào MySQL
├── library_framework/            # Code có dùng thư viện
│   ├── __pycache__/          
│   ├── charts/                   # Thư mục lưu trữ các hình ảnh biểu đồ 
│   ├── library_framework/        # Thư mục con bổ trợ cấu trúc framework
│   ├── app.py                    # File khởi chạy giao diện chính của ứng dụng Web Streamlit
│   ├── minh_truong_tab.py        # File thành phần giao diện (Phân đoạn Streamlit do Minh Trường phụ trách)
│   ├── EDA.ipynb                 # Notebook phân tích khám phá dữ liệu và vẽ biểu đồ xu hướng
│   ├── PCA.ipynb                 # Notebook áp dụng đại số tuyến tính giảm chiều dữ liệu
│   ├── Clustering.ipynb          # Notebook thuật toán Phân cụm không giám sát (K-Means) các trạng thái không khí
│   ├── Classification.ipynb      # Notebook huấn luyện mô hình Phân loại đa lớp cấp độ ô nhiễm
│   ├── Best_model.ipynb          # Notebook thực nghiệm song song, so sánh hiệu năng toán học 3 model để chọn Best Model dự đoán
│   ├── best_model.pkl            # Tệp tĩnh đóng gói toàn bộ cấu trúc và trọng số toán học tối ưu của XGBoost Regressor
│   ├── Time_Series_Forecast.ipynb # Notebook dự báo chuỗi thời gian đối chứng bằng Facebook Prophet
│   ├── Forecasting.py            # Module Python phục vụ tính toán các kịch bản dự báo tương lai
│   ├── Model Interpretation.ipynb # Notebook giải thích mô hình (SHAP/Feature Importance) phá vỡ hộp đen AI
│   ├── Ethical_Bias_Report.ipynb # Notebook rà soát, đánh giá định kiến và đạo đức của thuật toán
│   ├── Recommendation_System.py  # Hệ thống khuyến nghị theo nhóm người và ngữ cảnh
│   ├── context_advice_rules.csv  # Bảng data hệ khuyến nghị theo ngữ cảnh 
│   ├── recommendation_table.csv  # Bảng data hệ khuyến nghị theo nhóm người
│   ├── xgboost_best_model.json   # Tệp JSON lưu trữ cấu trúc cây quyết định đặc trưng của XGBoost Classification
│   └── xgboost_meta.json         # Tệp JSON lưu trữ siêu dữ liệu (metadata) cấu hình siêu tham số mô hình
├── manual_implementation/        # Code không dùng thư viện
│   └── Classification_manual.ipynb # Thử nghiệm tự lập trình thuật toán phân loại không dùng thư viện
├── .gitignore                    # Tệp cấu hình bỏ qua các file rác hệ thống, file .pkl, và file dữ liệu nặng
├── README.md                     # Tài liệu hướng dẫn cài đặt, vận hành dự án và nghiệm thu kết quả
└── requirements.txt              # Danh sách toàn bộ các thư viện cần cài đặt 

3. Ý nghĩa các cột
- Tổng cộng: 24 cột
  + local_time: Thời gian lấy dữ liệu (theo giờ Việt Nam)
  + date: Ngày thực hiện lấy dữ liệu
  + year / month: Năm / tháng lấy dữ liệu
  + hour: Giờ trong ngày (0h - 23h)
  + day_of_week: Thứ trong tuần (0:Thứ hai - 6:Chủ nhật)
  + is_weekend: Nếu cuối tuần thì 1 - ngày thường là 0
  + is_rush_hour: Nếu trong khung giờ tắc (6h-9h và 17h-20h) thì 1 - giờ thường thì 0
  + season: Mùa trong năm (0: Đông, 1: Xuân, 2: Hạ, 3: Thu)
  + aqi: Chỉ số chất lượng không khí tổng hợp
  + pm25 (µg/m³): Bụi mịn có đường kính < 2.5 micromet
  + pm10 (µg/m³): Bụi mịn có đường kính < 10 micromet
  + co (µg/m³) : Khí Carbon Monoxide (thường từ khí thải xe, công nghiệp) 
  + no2 (µg/m³): Khí Nitrogen Dioxide - chủ yếu từ đốt nhiên liệu (xe, nhà máy)
  + o3 (µg/m³): Khí Ozone tầng mặt đất
  + so2 (µg/m³): Khí Sulfur Dioxide - chủ yếu từ đốt than, dầu chứa lưu huỳnh (nhà máy, công nghiệp)
  + clouds (%): Độ che phủ mây. (Dùng để theo dõi mức độ bức xạ mặt trời; mây nhiều thường làm giảm quá trình hình thành Ozone tầng mặt đất).
  + precipitation (mm): Lượng mưa. (Đây là biến làm sạch không khí; mưa càng lớn AQI thường càng thấp do hiện tượng rửa trôi bụi mịn).
  + pressure (hPa): Áp suất khí quyển. (Áp suất thay đổi liên quan đến các đợt gió mùa; áp suất cao thường đi kèm với hiện tượng nghịch nhiệt khiến ô nhiễm bị kẹt lại sát mặt đất).
  + relative_humidity (%): Độ ẩm tương đối. (Độ ẩm cao ở Hà Nội thường gây ra sương mù, giữ chân bụi mịn PM2.5 khiến chỉ số ô nhiễm tăng vọt).
  + temperature (°C): Nhiệt độ môi trường. (Nhiệt độ ảnh hưởng đến tốc độ phản ứng hóa học của các chất khí và sự luân chuyển không khí theo chiều đứng).
  + uv_index: Chỉ số tia cực tím. (Liên quan trực tiếp đến phản ứng quang hóa tạo ra các chất ô nhiễm thứ cấp).
  + wind_speed (m/s): Tốc độ gió. (Biến số quan trọng để dự báo sự khuếch tán; gió càng mạnh thì bụi càng nhanh bị thổi đi nơi khác).

