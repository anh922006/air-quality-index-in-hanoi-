"""
Streamlit Dashboard — AQI Hà Nội 2022–2025
==========================================
Tab 1: EDA — Phân tích dữ liệu
Tab 2: Model Results — Kết quả mô hình
Tab 3: Recommendation — Hệ thống khuyến nghị
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AQI Hà Nội Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .aqi-good     { background: #00e400; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; }
    .aqi-moderate { background: #ffff00; color: black; padding: 6px 14px; border-radius: 20px; font-weight: bold; }
    .aqi-usg      { background: #ff7e00; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; }
    .aqi-unhealthy{ background: #ff0000; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; }
    .aqi-very     { background: #8f3f97; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; }
    .aqi-hazardous{ background: #7e0023; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    from sqlalchemy import create_engine
    engine = create_engine('mysql+pymysql://root:anh922006@localhost:3306/hanoi_aqi')
    df = pd.read_sql('SELECT * FROM aqi_data', con=engine)
    df.columns = df.columns.str.strip().str.lower()
    df['local_time'] = pd.to_datetime(df['local_time'])
    df = df.sort_values('local_time').reset_index(drop=True)
    return df

@st.cache_resource
def load_model(df):
    import xgboost as xgb
    FEATURES = [
        'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
        'clouds', 'precipitation', 'pressure',
        'relative_humidity', 'temperature', 'uv_index', 'wind_speed',
        'month', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season'
    ]
    train = df[df['year'] < 2025].copy()
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(train[FEATURES], train['aqi'])
    return model, FEATURES

with st.spinner("⏳ Đang kết nối MySQL và tải dữ liệu..."):
    try:
        df = load_data()
        model, FEATURES = load_model(df)
        st.success(f"✅ Đã tải {len(df):,} bản ghi từ MySQL")
    except Exception as e:
        st.error(f"❌ Lỗi kết nối MySQL: {e}")
        st.stop()

SEASON_MAP   = {0: 'Đông ❄️', 1: 'Xuân 🌸', 2: 'Hè ☀️', 3: 'Thu 🍂'}
WEATHER_COLS = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
                'clouds', 'precipitation', 'pressure',
                'relative_humidity', 'temperature', 'uv_index', 'wind_speed']

AQI_COLORS = {
    'Good':                          '#00e400',
    'Moderate':                      '#ffff00',
    'Unhealthy for sensitive groups':'#ff7e00',
    'Unhealthy':                     '#ff0000',
    'Very Unhealthy':                '#8f3f97',
    'Hazardous':                     '#7e0023',
}

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/32px-Flag_of_Vietnam.svg.png", width=32)
    st.title("AQI Hà Nội")
    st.caption("2022 – 2025 · Machine Learning")
    st.divider()

    st.subheader("📊 Tổng quan")
    st.metric("Tổng bản ghi", f"{len(df):,}")
    st.metric("AQI trung bình", f"{df['aqi'].mean():.1f}")
    st.metric("AQI cao nhất", f"{df['aqi'].max():.0f}")
    st.metric("Ngày Hazardous", f"{(df['aqi'] > 300).sum():,} giờ")
    st.divider()

    tab_select = st.radio(
        "Chọn tab",
        ["📈 EDA", "🤖 Model Results", "💡 Recommendation"],
        label_visibility="collapsed"
    )

# ══════════════════════════════════════════════════════
# TAB 1: EDA
# ══════════════════════════════════════════════════════
if tab_select == "📈 EDA":
    st.title("📈 Phân tích khám phá dữ liệu (EDA)")
    st.caption("Dữ liệu AQI Hà Nội theo giờ từ 2022–2025")

    # Row 1 — metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cat_counts = df['aqi_category'].value_counts()
    for col, cat, emoji in zip(
        [c1, c2, c3, c4, c5, c6],
        ['Good', 'Moderate', 'Unhealthy for sensitive groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
        ['🟢', '🟡', '🟠', '🔴', '🟣', '⚫']
    ):
        n = cat_counts.get(cat, 0)
        col.metric(f"{emoji} {cat.split()[0]}", f"{n:,}", f"{n/len(df)*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Biểu đồ 1: AQI theo giờ
        hourly = df.groupby('hour')['aqi'].mean().reset_index()
        fig = px.bar(hourly, x='hour', y='aqi',
                     title='AQI trung bình theo giờ trong ngày',
                     labels={'hour': 'Giờ', 'aqi': 'AQI trung bình'},
                     color='aqi', color_continuous_scale='RdYlGn_r')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Biểu đồ 2: Boxplot theo mùa
        df['season_name'] = df['season'].map({0: 'Đông', 1: 'Xuân', 2: 'Hè', 3: 'Thu'})
        fig = px.box(df, x='season_name', y='aqi',
                     title='Phân phối AQI theo mùa',
                     labels={'season_name': 'Mùa', 'aqi': 'AQI'},
                     color='season_name',
                     category_orders={'season_name': ['Đông', 'Xuân', 'Hè', 'Thu']},
                     color_discrete_map={'Đông': '#5B9BD5', 'Xuân': '#70AD47', 'Hè': '#FF0000', 'Thu': '#ED7D31'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Biểu đồ 3: Trend AQI theo tháng 2022-2025
        monthly = df.groupby(['year', 'month'])['aqi'].mean().reset_index()
        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        fig = px.line(monthly, x='date', y='aqi', color='year',
                      title='Xu hướng AQI trung bình theo tháng (2022–2025)',
                      labels={'date': 'Tháng', 'aqi': 'AQI', 'year': 'Năm'})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Biểu đồ 4: Tỷ lệ AQI Category
        cat_df = df['aqi_category'].value_counts().reset_index()
        cat_df.columns = ['category', 'count']
        fig = px.pie(cat_df, names='category', values='count',
                     title='Tỷ lệ các mức AQI',
                     color='category',
                     color_discrete_map=AQI_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    # Biểu đồ 5: Heatmap tương quan
    st.subheader("🔥 Heatmap tương quan 13 features")
    feat13 = ['aqi', 'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
              'clouds', 'precipitation', 'pressure',
              'relative_humidity', 'temperature', 'uv_index', 'wind_speed']
    corr = df[feat13].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect='auto',
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title='Ma trận tương quan')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Biểu đồ 6: Rush hour
    st.subheader("🚦 AQI giờ cao điểm vs không cao điểm theo mùa")
    rush_df = df.groupby(['season_name', 'is_rush_hour'])['aqi'].mean().reset_index()
    rush_df['Loại giờ'] = rush_df['is_rush_hour'].map({0: 'Không cao điểm', 1: 'Giờ cao điểm'})
    fig = px.bar(rush_df, x='season_name', y='aqi', color='Loại giờ',
                 barmode='group',
                 labels={'season_name': 'Mùa', 'aqi': 'AQI trung bình'},
                 category_orders={'season_name': ['Đông', 'Xuân', 'Hè', 'Thu']})
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 2: MODEL RESULTS
# ══════════════════════════════════════════════════════
elif tab_select == "🤖 Model Results":
    st.title("🤖 Kết quả mô hình Machine Learning")

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb

    train = df[df['year'] < 2025].copy()
    test  = df[df['year'] == 2025].copy()
    X_train, y_train = train[FEATURES], train['aqi']
    X_test,  y_test  = test[FEATURES],  test['aqi']

    @st.cache_data
    def run_models(_X_train, _y_train, _X_test, _y_test):
        lr = LinearRegression().fit(_X_train, _y_train)
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1).fit(_X_train, _y_train)
        xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbosity=0).fit(_X_train, _y_train)
        results = []
        for name, m in [('Linear Regression', lr), ('Random Forest', rf), ('XGBoost', xgb_m)]:
            yp = m.predict(_X_test)
            results.append({
                'Model': name,
                'RMSE':  round(np.sqrt(mean_squared_error(_y_test, yp)), 2),
                'MAE':   round(mean_absolute_error(_y_test, yp), 2),
                'R²':    round(r2_score(_y_test, yp), 4),
                'preds': yp
            })
        return results, lr, rf, xgb_m

    with st.spinner("⏳ Đang train 3 models..."):
        results, lr, rf, xgb_m = run_models(X_train, y_train, X_test, y_test)

    # Bảng so sánh
    st.subheader("📊 Bảng so sánh 3 model")
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'preds'} for r in results])
    best_idx = res_df['RMSE'].idxmin()

    styled = res_df.style.highlight_min(subset=['RMSE', 'MAE'], color='#d4edda') \
                         .highlight_max(subset=['R²'], color='#d4edda')
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.success(f"🏆 Best model: **{res_df.loc[best_idx, 'Model']}** (RMSE={res_df.loc[best_idx, 'RMSE']})")

    col1, col2 = st.columns(2)

    with col1:
        # Predicted vs Actual theo tháng
        test_plot = test.copy()
        for r in results:
            test_plot[r['Model']] = r['preds']
        monthly = test_plot.groupby('month').agg(
            Actual=('aqi', 'mean'),
            **{r['Model']: (r['Model'], 'mean') for r in results}
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['Actual'],
                                  mode='lines+markers', name='Thực tế',
                                  line=dict(color='black', width=2)))
        colors = ['#D85A30', '#7F77DD', '#1D9E75']
        for r, c in zip(results, colors):
            fig.add_trace(go.Scatter(x=monthly['month'], y=monthly[r['Model']],
                                      mode='lines+markers', name=r['Model'],
                                      line=dict(dash='dash', color=c)))
        fig.update_layout(title='Predicted vs Actual theo tháng (2025)',
                          xaxis_title='Tháng', yaxis_title='AQI')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter best model
        best = results[best_idx]
        fig = px.scatter(x=y_test, y=best['preds'], opacity=0.3,
                         labels={'x': 'AQI thực tế', 'y': f'AQI dự báo'},
                         title=f"Scatter – {best['Model']} (R²={best['R²']})",
                         color_discrete_sequence=['#D85A30'])
        mn, mx = min(y_test.min(), best['preds'].min()), max(y_test.max(), best['preds'].max())
        fig.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(color='black', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance XGBoost
    st.subheader("🔍 Feature Importance (XGBoost)")
    fi = pd.DataFrame({'Feature': FEATURES, 'Importance': xgb_m.feature_importances_})
    fi = fi.sort_values('Importance', ascending=True).tail(15)
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Blues',
                 title='Top 15 Features quan trọng nhất')
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 3: RECOMMENDATION
# ══════════════════════════════════════════════════════
elif tab_select == "💡 Recommendation":
    st.title("💡 Hệ thống khuyến nghị AQI thông minh")
    st.caption("Nhập ngày và giờ để nhận dự báo AQI và lời khuyên phù hợp")

    def get_season(month):
        if month in [12, 1, 2]:  return 0
        elif month in [3, 4, 5]: return 1
        elif month in [6, 7, 8]: return 2
        else:                     return 3

    def get_is_rush_hour(hour):
        return 1 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0

    def get_aqi_level(v):
        if v <= 50:    return 'Good'
        elif v <= 100: return 'Moderate'
        elif v <= 150: return 'Unhealthy for sensitive groups'
        elif v <= 200: return 'Unhealthy'
        elif v <= 300: return 'Very Unhealthy'
        else:          return 'Hazardous'

    AQI_RULES = {
        'Good':                          {'emoji': '🟢', 'color': '#00e400', 'Trẻ em': '✅ Hoạt động bình thường', 'Người già': '✅ Hoạt động bình thường', 'Bệnh hô hấp': '✅ Hoạt động bình thường', 'Người khỏe mạnh': '✅ Hoạt động bình thường'},
        'Moderate':                      {'emoji': '🟡', 'color': '#e6e600', 'Trẻ em': '⚠️ Hạn chế vận động mạnh', 'Người già': '⚠️ Tránh vận động mạnh', 'Bệnh hô hấp': '⚠️ Theo dõi triệu chứng', 'Người khỏe mạnh': '✅ Hoạt động bình thường'},
        'Unhealthy for sensitive groups':{'emoji': '🟠', 'color': '#ff7e00', 'Trẻ em': '🔶 Đeo khẩu trang khi ra ngoài', 'Người già': '🔶 Đeo khẩu trang N95', 'Bệnh hô hấp': '🔴 Tránh ra ngoài', 'Người khỏe mạnh': '⚠️ Hạn chế vận động mạnh'},
        'Unhealthy':                     {'emoji': '🔴', 'color': '#ff0000', 'Trẻ em': '🔴 Không nên ra ngoài', 'Người già': '🔴 Ở trong nhà', 'Bệnh hô hấp': '🔴 Ở trong nhà hoàn toàn', 'Người khỏe mạnh': '🔶 Đeo N95 nếu bắt buộc ra ngoài'},
        'Very Unhealthy':                {'emoji': '🟣', 'color': '#8f3f97', 'Trẻ em': '🚨 Ở trong nhà hoàn toàn', 'Người già': '🚨 Ở trong nhà hoàn toàn', 'Bệnh hô hấp': '🚨 Chuẩn bị thuốc cấp cứu', 'Người khỏe mạnh': '🔴 Tránh mọi hoạt động ngoài trời'},
        'Hazardous':                     {'emoji': '⚫', 'color': '#7e0023', 'Trẻ em': '🚨 Cấm ra ngoài', 'Người già': '🚨 Liên hệ y tế ngay', 'Bệnh hô hấp': '🚨 Cần hỗ trợ y tế khẩn cấp', 'Người khỏe mạnh': '🚨 Không ra ngoài'},
    }

    # Input form
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        date_input = st.date_input("📅 Chọn ngày", value=pd.Timestamp.today())
    with col2:
        hour_input = st.slider("⏰ Chọn giờ", 0, 23, 8)
    with col3:
        st.write("")
        st.write("")
        predict_btn = st.button("🔍 Dự báo AQI", use_container_width=True, type="primary")

    if predict_btn:
        target_date = pd.Timestamp(date_input)
        month       = target_date.month
        season      = get_season(month)
        is_weekend  = 1 if target_date.dayofweek >= 5 else 0
        is_rush     = get_is_rush_hour(hour_input)

        mask = (df['month'] == month) & (df['hour'] == hour_input)
        if mask.sum() == 0:
            mask = df['month'] == month
        avg_weather = df[mask][WEATHER_COLS].mean().to_dict()

        user_input = {
            **avg_weather,
            'month': month, 'hour': hour_input,
            'day_of_week': target_date.dayofweek,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush,
            'season': season,
        }

        pred_aqi = model.predict(pd.DataFrame([user_input])[FEATURES])[0]
        level    = get_aqi_level(round(pred_aqi))
        rec      = AQI_RULES[level]

        # Hiển thị kết quả
        st.divider()
        dow_names = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']
        season_labels = {0:'Đông ❄️', 1:'Xuân 🌸', 2:'Hè ☀️', 3:'Thu 🍂'}

        st.subheader(f"📍 {date_input.strftime('%d/%m/%Y')} — {hour_input:02d}:00  |  {dow_names[target_date.dayofweek]}  |  {season_labels[season]}")
        st.markdown(f"{'🚦 Giờ cao điểm' if is_rush else '🕐 Không cao điểm'}  |  {'🎉 Cuối tuần' if is_weekend else '📅 Ngày thường'}")

        # AQI result
        col_aqi, col_weather = st.columns([1, 2])
        with col_aqi:
            st.markdown(f"""
            <div style="background:{rec['color']}; border-radius:12px; padding:24px; text-align:center; color:white;">
                <div style="font-size:40px">{rec['emoji']}</div>
                <div style="font-size:48px; font-weight:bold">{pred_aqi:.1f}</div>
                <div style="font-size:16px">{level}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_weather:
            st.markdown("**🌡️ Điều kiện thời tiết dự kiến**")
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Nhiệt độ", f"{avg_weather['temperature']:.1f}°C")
            wc2.metric("Độ ẩm", f"{avg_weather['relative_humidity']:.0f}%")
            wc3.metric("Gió", f"{avg_weather['wind_speed']:.1f} m/s")
            wc1.metric("PM2.5", f"{avg_weather['pm25']:.1f}")
            wc2.metric("PM10", f"{avg_weather['pm10']:.1f}")
            wc3.metric("NO2", f"{avg_weather['no2']:.1f}")

        st.divider()

        # Khuyến nghị theo nhóm
        st.subheader("👥 Khuyến nghị theo nhóm người dùng")
        gc1, gc2, gc3, gc4 = st.columns(4)
        for col, group in zip([gc1, gc2, gc3, gc4],
                               ['Trẻ em', 'Người già', 'Bệnh hô hấp', 'Người khỏe mạnh']):
            col.info(f"**{group}**\n\n{rec[group]}")

        # Context-aware advice
        st.subheader("💡 Lời khuyên thông minh")
        tips = []
        if is_rush and not is_weekend:
            if pred_aqi > 100:
                tips.append("🚗 Giờ cao điểm + ô nhiễm cao: tránh Nguyễn Trãi, Tây Sơn, Giải Phóng")
                tips.append("🚇 Nên dùng metro/xe buýt thay xe máy")
            else:
                tips.append("🚗 Giờ cao điểm: đeo khẩu trang khi đi đường")

        if month in [12, 1, 2] and pred_aqi > 100:
            tips.append("❄️ Mùa Đông: nghịch nhiệt khiến bụi mịn tích tụ — AQI xấu nhất năm")
        elif month in [6, 7, 8]:
            tips.append("☀️ Mùa Hè: O3 và UV Index cao — tránh ra ngoài lúc 11h–15h")
        elif month in [3, 4, 5]:
            tips.append("🌸 Mùa Xuân: độ ẩm cao, sương mù nhiều — bụi mịn dễ tích tụ")

        if 0 <= hour_input <= 4 and pred_aqi > 100:
            tips.append("🌙 Ban đêm AQI cao: bật máy lọc không khí chế độ im lặng khi ngủ")

        general = {
            'Good': "🍃 Không khí trong lành: tăng cường thông gió, mở cửa sổ",
            'Moderate': "🌤️ AQI trung bình: có thể mở cửa sổ, hạn chế nếu nhà gần đường lớn",
            'Unhealthy for sensitive groups': "🏠 Nên đóng bớt cửa sổ, dùng máy lọc không khí",
            'Unhealthy': "🚫 Đóng chặt cửa sổ, bật máy lọc không khí liên tục",
            'Very Unhealthy': "🚨 Đóng kín toàn bộ cửa sổ, tránh mọi khe hở thông gió",
            'Hazardous': "🚨 Cực kỳ nguy hiểm — đóng kín nhà cửa, lọc khí tối đa"
        }
        tips.append(general[level])

        for tip in tips:
            st.markdown(f"→ {tip}")

        # AQI gauge chart
        st.divider()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_aqi, 1),
            title={'text': "Chỉ số AQI dự báo"},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': rec['color']},
                'steps': [
                    {'range': [0, 50],   'color': '#00e400'},
                    {'range': [50, 100],  'color': '#ffff00'},
                    {'range': [100, 150], 'color': '#ff7e00'},
                    {'range': [150, 200], 'color': '#ff0000'},
                    {'range': [200, 300], 'color': '#8f3f97'},
                    {'range': [300, 500], 'color': '#7e0023'},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Lịch sử AQI cùng tháng/giờ
        st.subheader(f"📅 Lịch sử AQI tháng {month} lúc {hour_input:02d}:00")
        hist = df[mask][['local_time', 'aqi', 'aqi_category']].copy()
        hist['Ngày'] = hist['local_time'].dt.strftime('%d/%m/%Y')
        fig = px.scatter(hist, x='local_time', y='aqi', color='aqi_category',
                         color_discrete_map=AQI_COLORS,
                         labels={'local_time': 'Thời gian', 'aqi': 'AQI'},
                         title=f"AQI lịch sử tháng {month} lúc {hour_input:02d}:00")
        fig.add_hline(y=pred_aqi, line_dash="dash", line_color="red",
                      annotation_text=f"Dự báo: {pred_aqi:.1f}")
        st.plotly_chart(fig, use_container_width=True)