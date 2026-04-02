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
├── data_raw/             # 4 file CSV gốc (2022, 2023, 2024, 2025)
├── clean/                
│   ├── clean.ipynb       # Code gộp file, nội suy Missing values & xử lý Outliers
│   └── hanoi_aqi_cleaned.csv # Dataset sạch để chạy Model
├── manual_implementation/  # Triển khai thuật toán Không dùng thư viện
│   └── Classification_manual.ipynb  
├── library_framework/    # Triển khai bằng thư viện
│   └── PCA.ipynb         
├── .gitignore            # Bỏ qua các file rác và file data quá lớn
├── README.md             # Tài liệu hướng dẫn dự án 
└── requirements.txt      # Thư viện cần cài đặt

3. Ý nghĩa các cột
- Tổng cộng: 26 cột
  + local_time: Thời gian lấy dữ liệu (theo giờ Việt Nam)
  + date: Ngày thực hiện lấy dữ liệu
  + year / month: Năm / tháng lấy dữ liệu
  + hour: Giờ trong ngày (0h - 23h)
  + day_of_week: Thứ trong tuần (0:Thứ hai - 6:Chủ nhật)
  + is_weekend: Nếu cuối tuần thì 1 - ngày thường là 0
  + season: Mùa trong năm (0: Đông, 1: Xuân, 2: Hạ, 3: Thu)
  + aqi: Chỉ số chất lượng không khí tổng hợp
  + pm25 (µg/m³): Bụi mịn có đường kính < 2.5 micromet
  + pm10 (µg/m³): Bụi mịn có đường kính < 10 micromet
  + co (µg/m³) : Khí Carbon Monoxide (thường từ khí thải xe, công nghiệp) 
  + no2 (µg/m³): Khí Nitrogen Dioxide - chủ yếu từ đốt nhiên liệu (xe, nhà máy)
  + o3 (µg/m³): Khí Ozone tầng mặt đất
  + so2 (µg/m³): Khí Sulfur Dioxide - chủ yếu từ đốt than, dầu chứa lưu huỳnh (nhà máy, công nghiệp)

