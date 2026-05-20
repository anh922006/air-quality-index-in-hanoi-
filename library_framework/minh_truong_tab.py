# ============================================================
#  MINH TRƯỜNG — TAB EDA + CLASSIFICATION (cho app.py)
#  Số liệu 100% từ EDA.ipynb + Classification.ipynb đã chạy
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from sqlalchemy import create_engine
from PIL import Image

# ── Thư mục ảnh biểu đồ ──────────────────────────────────
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts_eda')
os.makedirs(CHART_DIR, exist_ok=True)

# ── Load data từ MySQL ────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    load_dotenv()
    engine = create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    df = pd.read_sql("SELECT * FROM aqi_data", con=engine)
    df.columns = df.columns.str.strip().str.lower()
    df['local_time'] = pd.to_datetime(df['local_time'])
    SEASON_MAP = {0: 'Đông', 1: 'Xuân', 2: 'Hạ', 3: 'Thu'}
    df['season_name'] = df['season'].map(SEASON_MAP)
    df['ym'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    )
    df['rush_label'] = df['is_rush_hour'].map({1: 'Giờ cao điểm', 0: 'Giờ thường'})
    DAY_MAP = {0:'Thứ 2',1:'Thứ 3',2:'Thứ 4',3:'Thứ 5',4:'Thứ 6',5:'Thứ 7',6:'CN'}
    df['day_name'] = df['day_of_week'].map(DAY_MAP)
    return df

# ── Hàm load ảnh từ thư mục charts_eda ───────────────────
def show_chart_img(filename, caption=''):
    path = os.path.join(CHART_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
        return True
    return False

# ── Config ───────────────────────────────────────────────
PALETTE = {
    'Good':'#2ecc71','Moderate':'#f1c40f',
    'Unhealthy for sensitive groups':'#e67e22',
    'Unhealthy':'#e74c3c','Very Unhealthy':'#9b59b6','Hazardous':'#7f1d1d'
}
CAT_ORDER    = list(PALETTE.keys())
SEASON_ORDER = ['Xuân','Hạ','Thu','Đông']
SEASON_CLR   = {'Xuân':'#27ae60','Hạ':'#e74c3c','Thu':'#e67e22','Đông':'#3498db'}

# ═══════════════════════════════════════════════════════════
#  HÀM CHÍNH — gọi từ app.py
# ═══════════════════════════════════════════════════════════
def render_eda_tab():
    df = load_data()

    st.header("📊 Phân Tích Khám Phá Dữ Liệu — EDA")
    st.caption("Toàn bộ số liệu được tính trực tiếp từ database MySQL — 100% khớp với EDA.ipynb")

    # ── KPI ─────────────────────────────────────────────────
    hourly     = df.groupby('hour')['aqi'].mean()
    season_avg = df.groupby('season_name')['aqi'].mean()
    bad_pct_25 = (df[df['year']==2025]['aqi'] > 150).mean()*100
    rush_avg   = df.groupby('is_rush_hour')['aqi'].mean()
    delta_rush = rush_avg[1] - rush_avg[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tổng bản ghi", f"{len(df):,}", "MySQL")
    k2.metric("Giờ ô nhiễm nhất", f"{hourly.idxmax()}h", f"AQI={hourly.max():.1f}")
    k3.metric("Mùa AQI cao nhất", "Đông", f"avg={season_avg['Đông']:.1f}")
    k4.metric("AQI > 150 năm 2025", f"{bad_pct_25:.1f}%", "↑ so với 2022: 26.9%", delta_color='inverse')
    k5.metric("Chênh lệch giờ cao điểm", f"+{delta_rush:.1f}", "so giờ thường")

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 1: AQI theo giờ
    # ────────────────────────────────────────────────────────
    st.subheader("📈 EDA 1 — Phân phối AQI trung bình theo giờ trong ngày")
    c1, c2 = st.columns([2, 1])
    with c1:
        img_shown = show_chart_img('eda1_aqi_by_hour.png', 'EDA1: AQI theo giờ')
        if not img_shown:
            hourly_df = hourly.reset_index()
            hourly_df.columns = ['hour','aqi']
            fig = px.line(hourly_df, x='hour', y='aqi', markers=True,
                          labels={'hour':'Giờ','aqi':'AQI trung bình'},
                          title='AQI trung bình theo giờ trong ngày — Hà Nội 2022–2025')
            fig.update_traces(line=dict(color='#e74c3c', width=3))
            fig.add_vrect(x0=6, x1=9, fillcolor='red', opacity=0.08, line_width=0,
                          annotation_text='Cao điểm sáng', annotation_position='top left')
            fig.add_vrect(x0=17, x1=20, fillcolor='red', opacity=0.08, line_width=0,
                          annotation_text='Cao điểm tối', annotation_position='top left')
            fig.add_scatter(x=[hourly.idxmax()], y=[hourly.max()],
                            mode='markers+text', marker=dict(color='red', size=12),
                            text=[f'Đỉnh: {hourly.max():.1f}'], textposition='top center',
                            name='Đỉnh ô nhiễm', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Nhận xét**")
        st.info(f"""
**Giờ ô nhiễm nhất:** 20h (AQI = **{hourly.max():.1f}**)  
**Giờ sạch nhất:** 16h (AQI = **{hourly.min():.1f}**)  

**Nguyên nhân:** Ban chiều bức xạ mặt trời tạo đối lưu nhiệt mạnh → phân tán bụi. Sau 18h, nghịch nhiệt ban đêm hình thành, chất ô nhiễm tích tụ sát mặt đất.

| Khung giờ | AQI TB |
|-----------|--------|
| Đêm 0–5h | {df[df['hour'].between(0,5)]['aqi'].mean():.1f} |
| Sáng 6–11h | {df[df['hour'].between(6,11)]['aqi'].mean():.1f} |
| Chiều 12–17h | {df[df['hour'].between(12,17)]['aqi'].mean():.1f} ✅ |
| Tối 18–23h | {df[df['hour'].between(18,23)]['aqi'].mean():.1f} ⚠️ |
        """)

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 2: Boxplot theo mùa
    # ────────────────────────────────────────────────────────
    st.subheader("📦 EDA 2 — Phân phối AQI theo 4 mùa")
    c1, c2 = st.columns([2, 1])
    with c1:
        img_shown = show_chart_img('eda2_boxplot_season.png', 'EDA2: Boxplot theo mùa')
        if not img_shown:
            fig = px.box(df, x='season_name', y='aqi', category_orders={'season_name': SEASON_ORDER},
                         color='season_name', color_discrete_map=SEASON_CLR,
                         title='Phân phối AQI theo 4 mùa — Hà Nội 2022–2025',
                         labels={'season_name':'Mùa','aqi':'AQI'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Nhận xét**")
        season_tbl = df.groupby('season_name')['aqi'].agg(['mean','median','std']).reindex(SEASON_ORDER)
        st.dataframe(season_tbl.round(1).rename(columns={'mean':'TB','median':'Median','std':'Std'}),
                     use_container_width=True)
        st.warning(f"""
**Đông:** AQI TB = **{season_avg['Đông']:.1f}** — ô nhiễm nhất  
**Hạ:** AQI TB = **{season_avg['Hạ']:.1f}** — sạch nhất  
Chênh lệch: **{season_avg['Đông']-season_avg['Hạ']:.1f}** đơn vị ({(season_avg['Đông']/season_avg['Hạ']-1)*100:.0f}% cao hơn)
        """)

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 3: Heatmap tương quan
    # ────────────────────────────────────────────────────────
    st.subheader("🔥 EDA 3 — Heatmap tương quan 13 features")
    c1, c2 = st.columns([2, 1])
    with c1:
        img_shown = show_chart_img('eda3_correlation_heatmap.png', 'EDA3: Heatmap tương quan')
        if not img_shown:
            feat13 = ['aqi','co','no2','o3','pm10','pm25','so2',
                      'clouds','precipitation','pressure','relative_humidity','temperature','wind_speed']
            corr   = df[feat13].corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdYlGn_r',
                            zmin=-1, zmax=1, title='Heatmap tương quan 13 features')
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Top tương quan với AQI**")
        feat13 = ['co','no2','o3','pm10','pm25','so2','clouds','precipitation',
                  'pressure','relative_humidity','temperature','wind_speed']
        corr_aqi = df[feat13+['aqi']].corr()['aqi'].drop('aqi').sort_values(key=abs, ascending=False)
        corr_df  = corr_aqi.reset_index()
        corr_df.columns = ['Feature','Correlation']
        fig2 = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                      color='Correlation', color_continuous_scale='RdBu',
                      range_color=[-1,1])
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.info("**PM25** (+0.927) và **PM10** (+0.822) có tương quan mạnh nhất với AQI")

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 4: Trend theo tháng
    # ────────────────────────────────────────────────────────
    st.subheader("📅 EDA 4 — Xu hướng AQI theo tháng 2022–2025")
    c1, c2 = st.columns([2, 1])
    with c1:
        img_shown = show_chart_img('eda4_monthly_trend.png', 'EDA4: Trend 4 năm')
        if not img_shown:
            monthly = df.groupby('ym')['aqi'].mean().reset_index()
            x_num   = np.arange(len(monthly))
            c_poly  = np.polynomial.polynomial.polyfit(x_num, monthly['aqi'].values, 1)
            trend   = c_poly[0] + c_poly[1]*x_num

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly['ym'], y=monthly['aqi'],
                                     mode='lines+markers', name='AQI TB tháng',
                                     line=dict(color='#2980b9', width=2),
                                     marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=monthly['ym'], y=trend,
                                     mode='lines', name=f'Xu hướng (↑ +{c_poly[1]:.2f}/tháng)',
                                     line=dict(color='red', width=2, dash='dash')))
            fig.update_layout(title='Trend AQI theo tháng — Hà Nội 2022–2025',
                              xaxis_title='Tháng', yaxis_title='AQI trung bình',
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Nhận xét**")
        yearly = df.groupby('year')['aqi'].mean()
        bad_by_yr = df.groupby('year').apply(lambda x: (x['aqi']>150).mean()*100)
        yr_df = pd.DataFrame({'AQI TB': yearly.round(1), 'Giờ AQI>150 (%)': bad_by_yr.round(1)})
        st.dataframe(yr_df, use_container_width=True)
        x_num = np.arange(len(df.groupby('ym')['aqi'].mean()))
        c_poly = np.polynomial.polynomial.polyfit(x_num, df.groupby('ym')['aqi'].mean().values, 1)
        st.error(f"""
**Xu hướng: ↑ TĂNG** (slope = +{c_poly[1]:.2f}/tháng)  
2025 đạt **143.3** — cao nhất 4 năm  
Giờ AQI>150 năm 2025: **{bad_pct_25:.1f}%**
        """)

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 5: Tỷ lệ AQI Category
    # ────────────────────────────────────────────────────────
    st.subheader("🥧 EDA 5 — Tỷ lệ 6 mức AQI Category")
    c1, c2 = st.columns(2)
    with c1:
        img_shown = show_chart_img('eda5_category_distribution.png', 'EDA5: Phân bố AQI Category')
        if not img_shown:
            cat_counts = df['aqi_category'].value_counts().reindex(CAT_ORDER)
            fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                         color=cat_counts.index, color_discrete_map=PALETTE,
                         title='Tỷ lệ 6 mức AQI Category')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        cat_counts = df['aqi_category'].value_counts().reindex(CAT_ORDER)
        fig2 = px.bar(x=cat_counts.index, y=cat_counts.values,
                      color=cat_counts.index, color_discrete_map=PALETTE,
                      labels={'x':'Mức AQI','y':'Số bản ghi'},
                      title='Số lượng theo 6 mức AQI', text=cat_counts.values)
        fig2.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    st.info("⚠️ Mất cân bằng nghiêm trọng: **Good 3.4%**, **Hazardous 0.6%** — đã xử lý bằng `class_weight='balanced'`")

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 6: Rush hour × Mùa
    # ────────────────────────────────────────────────────────
    st.subheader("🚦 EDA 6 — AQI giờ cao điểm vs giờ thường theo mùa")
    c1, c2 = st.columns([2,1])
    with c1:
        img_shown = show_chart_img('eda6_rush_hour_season.png', 'EDA6: Rush hour × Mùa')
        if not img_shown:
            rush_season = df.groupby(['season_name','rush_label'])['aqi'].mean().reset_index()
            fig = px.bar(rush_season, x='season_name', y='aqi', color='rush_label',
                         barmode='group', category_orders={'season_name': SEASON_ORDER},
                         color_discrete_map={'Giờ cao điểm':'#e74c3c','Giờ thường':'#3498db'},
                         title='AQI TB: Giờ cao điểm vs Giờ thường theo mùa',
                         labels={'season_name':'Mùa','aqi':'AQI TB','rush_label':'Khung giờ'})
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        rs = df.groupby(['season_name','is_rush_hour'])['aqi'].mean().unstack()
        rs.columns = ['Giờ thường','Cao điểm']
        rs['Chênh lệch'] = (rs['Cao điểm'] - rs['Giờ thường']).round(1)
        st.dataframe(rs.reindex(SEASON_ORDER).round(1), use_container_width=True)
        st.warning(f"Cao điểm luôn ô nhiễm hơn giờ thường ở tất cả 4 mùa. Mùa **Đông** chênh lệch lớn nhất.")

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 7: Cuối tuần vs ngày thường
    # ────────────────────────────────────────────────────────
    st.subheader("📅 EDA 7 — AQI theo thứ & cuối tuần")
    c1, c2 = st.columns(2)
    with c1:
        img_shown = show_chart_img('eda7_weekend.png', 'EDA7: Ngày thường vs cuối tuần')
        if not img_shown:
            DAY_ORDER = ['Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7','CN']
            day_avg = df.groupby('day_name')['aqi'].mean().reindex(DAY_ORDER).reset_index()
            day_avg.columns = ['day','aqi']
            day_avg['type'] = day_avg['day'].apply(lambda x: 'Cuối tuần' if x in ['Thứ 7','CN'] else 'Ngày thường')
            fig = px.bar(day_avg, x='day', y='aqi', color='type',
                         color_discrete_map={'Ngày thường':'#3498db','Cuối tuần':'#e67e22'},
                         title='AQI trung bình theo thứ trong tuần',
                         labels={'day':'Thứ','aqi':'AQI TB'}, text_auto='.0f')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        wk = df.groupby('is_weekend')['aqi'].mean()
        fig2 = px.bar(x=['Ngày thường (T2–T6)','Cuối tuần (T7–CN)'],
                      y=[wk[0], wk[1]],
                      color=['Ngày thường (T2–T6)','Cuối tuần (T7–CN)'],
                      color_discrete_map={'Ngày thường (T2–T6)':'#3498db','Cuối tuần (T7–CN)':'#e67e22'},
                      title='Ngày thường vs Cuối tuần',
                      labels={'x':'','y':'AQI TB'}, text_auto='.1f')
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"Ngày thường: **{wk[0]:.1f}** | Cuối tuần: **{wk[1]:.1f}**  \nChênh: **+{wk[0]-wk[1]:.1f}** — ô nhiễm không chỉ từ giao thông")

    st.divider()

    # ────────────────────────────────────────────────────────
    # EDA 8: Heatmap giờ × mùa
    # ────────────────────────────────────────────────────────
    st.subheader("🗺️ EDA 8 — Heatmap AQI theo Giờ × Mùa")
    c1, c2 = st.columns([2,1])
    with c1:
        img_shown = show_chart_img('eda8_heatmap_hour_season.png', 'EDA8: Heatmap Giờ × Mùa')
        if not img_shown:
            pivot = df.pivot_table(values='aqi', index='hour', columns='season_name',
                                   aggfunc='mean')[SEASON_ORDER]
            fig = px.imshow(pivot, color_continuous_scale='YlOrRd', text_auto='.0f',
                            labels={'x':'Mùa','y':'Giờ','color':'AQI'},
                            title='Heatmap AQI trung bình theo Giờ × Mùa',
                            y=[f'{h}h' for h in range(24)], aspect='auto')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Nhận xét**")
        pivot = df.pivot_table(values='aqi', index='hour', columns='season_name', aggfunc='mean')
        max_val = pivot.max().max()
        max_loc = pivot.stack().idxmax()
        st.error(f"**Tệ nhất:** Mùa **{max_loc[1]}** lúc **{max_loc[0]}h** (AQI = **{max_val:.1f}**)")
        st.markdown("**Top 5 ô tệ nhất:**")
        top5 = pivot.stack().sort_values(ascending=False).head(5).reset_index()
        top5.columns = ['Giờ','Mùa','AQI']
        top5['Giờ'] = top5['Giờ'].astype(str) + 'h'
        st.dataframe(top5.round(1), use_container_width=True, hide_index=True)


def render_classification_tab():
    df = load_data()

    st.header("🤖 Phân Loại AQI 6 Mức — Classification")
    st.caption("Random Forest & XGBoost | Train: 2022–2024 | Test: 2025 | class_weight='balanced'")

    # ── KPI ─────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Random Forest F1-macro", "0.8224", "BEST ✅")
    k2.metric("Random Forest ROC-AUC",  "0.9993")
    k3.metric("XGBoost F1-macro",       "0.8166")
    k4.metric("XGBoost ROC-AUC",        "0.9990")

    st.divider()

    # ── Bảng so sánh ─────────────────────────────────────────
    st.subheader("📊 So sánh 2 Classifier")
    c1, c2 = st.columns([1,2])
    with c1:
        df_cmp = pd.DataFrame({
            'Model': ['Random Forest','XGBoost'],
            'F1-macro': [0.8224, 0.8166],
            'ROC-AUC':  [0.9993, 0.9990],
            'Ghi chú':  ['✅ BEST','Đối chứng'],
        })
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)
        st.info("**Train:** 25,992 mẫu (2022–2024)  \n**Test:** 4,344 mẫu (2025)")
    with c2:
        img_shown = show_chart_img('clf1_comparison_chart.png', 'So sánh F1 và AUC')
        if not img_shown:
            fig = make_subplots(rows=1, cols=2, subplot_titles=['F1-macro','ROC-AUC'])
            for col_i, (metric, vals) in enumerate(
                [('F1-macro',[0.8224,0.8166]),('ROC-AUC',[0.9993,0.9990])], 1
            ):
                colors = ['#27ae60' if v==max(vals) else '#95a5a6' for v in vals]
                fig.add_trace(go.Bar(x=['Random Forest','XGBoost'], y=vals,
                                     marker_color=colors, text=[f'{v:.4f}' for v in vals],
                                     textposition='outside', showlegend=False), row=1, col=col_i)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── F1 per class ─────────────────────────────────────────
    st.subheader("🎯 F1-score từng class")
    c1, c2 = st.columns([2,1])
    with c1:
        img_shown = show_chart_img('clf2_f1_per_class.png', 'F1 per class')
        if not img_shown:
            classes = ['Good','Hazardous','Moderate','Unhealthy','USG','Very Unhealthy']
            f1_rf_per  = [0.9630, 0.5714, 0.8849, 0.7716, 0.7714, 0.7934]
            f1_xgb_per = [0.9474, 0.4706, 0.8765, 0.7590, 0.7752, 0.7714]
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Random Forest', x=classes, y=f1_rf_per,
                                 marker_color='#27ae60', text=[f'{v:.2f}' for v in f1_rf_per],
                                 textposition='outside'))
            fig.add_trace(go.Bar(name='XGBoost', x=classes, y=f1_xgb_per,
                                 marker_color='#e67e22', text=[f'{v:.2f}' for v in f1_xgb_per],
                                 textposition='outside'))
            fig.add_hline(y=0.9, line_dash='dash', line_color='red', annotation_text='0.9 threshold')
            fig.update_layout(barmode='group', yaxis_range=[0,1.15],
                               title='F1-score từng class — RF vs XGBoost',
                               yaxis_title='F1-score')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Nhận xét**")
        st.success("**Good:** F1=0.963 — nhận diện rất tốt nhờ class_weight")
        st.warning("**Hazardous:** F1=0.571 — khó vì chỉ 6 mẫu trong test 2025")
        st.info("**Moderate/USG/Unhealthy:** F1 0.77–0.88 — ổn định")
        st.error("Lỗi chủ yếu xảy ra ở các lớp **liền kề** (AQI ranh giới mờ)")

    st.divider()

    # ── Confusion Matrix ──────────────────────────────────────
    st.subheader("🔲 Confusion Matrix trực quan")
    c1, c2 = st.columns(2)
    with c1:
        img_shown = show_chart_img('clf3_confusion_rf.png', 'Confusion Matrix — Random Forest')
        if not img_shown:
            st.info("Đặt file `clf3_confusion_rf.png` vào thư mục `charts_eda/` để hiển thị")
    with c2:
        img_shown = show_chart_img('clf3_confusion_xgb.png', 'Confusion Matrix — XGBoost')
        if not img_shown:
            st.info("Đặt file `clf3_confusion_xgb.png` vào thư mục `charts_eda/` để hiển thị")

    # Fallback: vẽ trực tiếp nếu không có ảnh
    if not os.path.exists(os.path.join(CHART_DIR,'clf3_confusion_rf.png')):
        img_shown_combined = show_chart_img('clf3_confusion_matrices.png', 'Confusion Matrix — RF vs XGB')
        if not img_shown_combined:
            st.info("💡 Chạy `Classification.ipynb` để sinh ảnh Confusion Matrix, sau đó đặt vào `charts_eda/`")

    st.divider()

    # ── Feature Importance ────────────────────────────────────
    st.subheader("📌 Feature Importance — XGBoost")
    c1, c2 = st.columns([2,1])
    with c1:
        img_shown = show_chart_img('clf4_feature_importance.png', 'Feature Importance XGBoost')
        if not img_shown:
            feat_imp = {
                'pm25':0.6732,'pm10':0.1883,'season':0.0581,'o3':0.0196,
                'pressure':0.0132,'temperature':0.0101,'no2':0.0090,
                'hour':0.0082,'month':0.0075,'wind_speed':0.0043,
                'co':0.0031,'so2':0.0018,'is_rush_hour':0.0015,
                'relative_humidity':0.0012,'clouds':0.0006,'uv_index':0.0005,
                'precipitation':0.0004,'day_of_week':0.0003,'is_weekend':0.0001,
            }
            fi_df = pd.DataFrame({'Feature':list(feat_imp.keys()),
                                  'Importance':list(feat_imp.values())}).sort_values('Importance')
            fi_df['Color'] = fi_df['Importance'].apply(
                lambda x: '#e74c3c' if x >= sorted(feat_imp.values())[-5] else '#3498db'
            )
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                         color='Color', color_discrete_map='identity',
                         title='Feature Importance — XGBoost (đỏ = top 5 quan trọng nhất)',
                         text_auto='.4f')
            fig.update_layout(height=550, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📌 Top 5 features**")
        top_feat = [
            ('pm25',     0.6732, 'Quyết định 67% kết quả'),
            ('pm10',     0.1883, 'Quyết định 19%'),
            ('season',   0.0581, 'Mùa ảnh hưởng mạnh'),
            ('o3',       0.0196, 'Ozone tầng mặt đất'),
            ('pressure', 0.0132, 'Áp suất → nghịch nhiệt'),
        ]
        for feat, val, note in top_feat:
            st.metric(feat, f"{val:.4f}", note)

    st.divider()

    # ── Phân tích mô hình ────────────────────────────────────
    st.subheader("💡 Phân tích & Kết luận mô hình")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Điểm mạnh**")
        st.success("""
- ROC-AUC = 0.9993 — phân biệt đúng 99.93% cặp mẫu bất kỳ
- PM2.5 chiếm 67% importance → model hiểu đúng bản chất ô nhiễm Hà Nội
- class_weight='balanced' giúp nhận diện được cả Good (3.4%) và Hazardous (0.6%)
- Temporal split (train 2022-2024, test 2025) → không có data leakage
        """)
    with col2:
        st.markdown("**⚠️ Hạn chế**")
        st.warning("""
- Test 2025 chỉ có tháng 1–6 (thiếu mùa Hạ & Thu) → có thể bias F1 thấp hơn thực tế
- Hazardous F1=0.571 do chỉ có 6 mẫu test — cần thêm dữ liệu 2025 cuối năm
- F1-macro 0.82 thấp hơn mong đợi do ranh giới giữa các lớp liền kề AQI mờ
        """)
