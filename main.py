import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 페이지 및 탭 설정
st.set_page_config(
    page_title="영화 유형 나누기 (K-Means Clustering)",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 군집 분석)")
st.write("영화의 주요 흥행 및 상영 지표를 활용하여 정답(라벨) 없이 영화들을 선택한 수의 유형으로 묶어 분석합니다.")

# 2. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    total_raw_count = len(df)
    
    # 필수 열 숫자형 변환
    cols_to_num = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
    for c in cols_to_num:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # 첫 주 관객이 0이 아니고 결측치가 없는 항목 필터링
    valid_mask = (
        df['first_scrn'].notna() & (df['first_scrn'] > 0) &
        df['total_audi'].notna() & (df['total_audi'] > 0) &
        df['days_in_top10'].notna() &
        df['first_week_audi'].notna() & (df['first_week_audi'] > 0)
    )
    
    df_clean = df[valid_mask].copy()
    
    # 파생 속성 생성
    df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
    df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])
    df_clean['long_run_index'] = (df_clean['total_audi'] / df_clean['first_week_audi']).clip(upper=20)
    
    valid_count = len(df_clean)
    return df_clean, total_raw_count, valid_count

df, total_count, valid_count = load_and_preprocess_data()

# 전체 편수와 묶은 편수 한 줄 표기
st.info(f"📊 **전체 데이터 수**: {total_count}편 | **결측치 및 조건을 제외하고 분석에 사용된 영화 수**: {valid_count}편")

st.markdown("---")

# 3. 속성 선택 UI
feature_options = {
    "log_first_scrn": "스크린 수(상용로그)",
    "log_total_audi": "누적 관객(상용로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수"
}

col_feat, col_k = st.columns([2, 1])

with col_feat:
    st.subheader("⚙️ 묶는 데 사용할 속성 선택")
    selected_feature_keys = st.multiselect(
        "군집화(K-Means)에 사용할 속성을 2개 이상 선택하세요. (기본값: 4개 모두)",
        options=list(feature_options.keys()),
        default=list(feature_options.keys()),
        format_func=lambda x: feature_options[x]
    )

# 4. 묶음 수(k) 선택 UI (2~7, 기본값 3)
with col_k:
    st.subheader("🔢 묶음 수(K) 선택")
    n_clusters = st.slider(
        "묶음 수를 선택하세요",
        min_value=2,
        max_value=7,
        value=3,
        step=1
    )

if len(selected_feature_keys) < 2:
    st.warning("⚠️ 영화를 나누기 위해서는 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

# 5. K-Means 군집화 (선택 속성 및 묶음 수 적용)
X = df[selected_feature_keys]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
initial_
