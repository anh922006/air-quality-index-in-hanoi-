import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

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

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

df.to_sql(
    name='aqi_data',       
    con=engine,
    if_exists='replace',    
    index=False,
    chunksize=1000        
)

print(f" Đã đẩy {len(df):,} hàng vào MySQL table 'aqi_data'")