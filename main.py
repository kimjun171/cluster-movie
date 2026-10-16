import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 및 탭 설정 (제목 및 어울리는 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기 (K-Means Clustering)",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 군집 분석)")
st.write("영화의 주요 흥행 및 상영 지표를 활용하여 정답(라벨) 없이 영화들을 3가지 유형으로 묶어 분석합니다.")

# 2. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    total_raw_count = len(df)
    
    # 필수 열 숫자형 변환 및 결측치/조건 처리
    cols_to_num = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']
    for c in cols_to_num:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # 첫 주 관객이 0이 아니고, 필수 값이 모두 존재하며 0보다 큰 영화만 필터링 (로그 변환을 위해 > 0)
    valid_mask = (
        df['first_scrn'].notna() & (df['first_scrn'] > 0) &
        df['total_audi'].notna() & (df['total_audi'] > 0) &
        df['days_in_top10'].notna() &
        df['first_week_audi'].notna() & (df['first_week_audi'] > 0)
    )
    
    df_clean = df[valid_mask].copy()
    
    # 파생 속성 생성
    # 1) 스크린 수 상용로그
    df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
    # 2) 누적 관객 상용로그
    df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])
    # 3) 100위권 일수 (기존 가공) -> days_in_top10 그대로
    # 4) 롱런 지수 = 누적 관객 / 첫 주 관객 (20 초과는 20으로 클리핑)
    df_clean['long_run_index'] = (df_clean['total_audi'] / df_clean['first_week_audi']).clip(upper=20)
    
    valid_count = len(df_clean)
    return df, df_clean, total_raw_count, valid_count

raw_df, df, total_count, valid_count = load_and_preprocess_data()

# 전체 편수와 묶은 편수를 한 줄로 안내
st.info(f"📊 **전체 데이터 수**: {total_count}편 | **결측치 및 조건을 제외하고 분석에 사용된 영화 수**: {valid_count}편")

st.markdown("---")

# 3. 속성 선택 UI (기본은 4개 모두 선택)
feature_options = {
    "log_first_scrn": "스크린 수(상용로그)",
    "log_total_audi": "누적 관객(상용로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수"
}

st.subheader("⚙️ 묶는 데 사용할 속성 선택")
selected_feature_keys = st.multiselect(
    "군집화(K-Means)에 사용할 속성을 2개 이상 선택하세요. (기본값: 4개 모두)",
    options=list(feature_options.keys()),
    default=list(feature_options.keys()),
    format_func=lambda x: feature_options[x]
)

if len(selected_feature_keys) < 2:
    st.warning("⚠️ 영화를 나누기 위해서는 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

# 4. K-Means 군집화 (StandardScaler 적용, 난수 고정 random_state=42)
X = df[selected_feature_keys]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
initial_labels = kmeans.fit_predict(X_scaled)

df['temp_cluster'] = initial_labels

# 5. 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 군집 이름 부여
cluster_avg_audi = df.groupby('temp_cluster')['total_audi'].mean().sort_values(ascending=False)
label_map = {old_cluster: new_symbol for old_cluster, new_symbol in zip(cluster_avg_audi.index, ['㉮', '㉯', '㉰'])}

df['cluster'] = df['temp_cluster'].map(label_map)
# 카테고리 순서 정렬
cluster_order = ['㉮', '㉯', '㉰']
df['cluster'] = pd.Categorical(df['cluster'], categories=cluster_order, ordered=True)

# 6. 2차원 산점도
st.markdown("---")
st.subheader("📌 2차원 산점도")

col_2d_x, col_2d_y = st.columns(2)

with col_2d_x:
    x_axis_2d = st.selectbox(
        "가로축(X축) 속성 선택",
        options=selected_feature_keys,
        index=0,
        format_func=lambda x: feature_options[x],
        key="2d_x"
    )

with col_2d_y:
    y_axis_2d = st.selectbox(
        "세로축(Y축) 속성 선택",
        options=selected_feature_keys,
        index=1 if len(selected_feature_keys) > 1 else 0,
        format_func=lambda x: feature_options[x],
        key="2d_y"
    )

fig_2d = px.scatter(
    df,
    x=x_axis_2d,
    y=y_axis_2d,
    color='cluster',
    category_orders={'cluster': cluster_order},
    hover_name='movieNm',
    hover_data={'total_audi': ':,', 'first_scrn': ':,', 'days_in_top10': True, 'long_run_index': ':.2f'},
    title=f"2차원 산점도 ({feature_options[x_axis_2d]} vs {feature_options[y_axis_2d]})",
    labels={
        x_axis_2d: feature_options[x_axis_2d],
        y_axis_2d: feature_options[y_axis_2d],
        'cluster': '영화 묶음'
    },
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig_2d.update_traces(marker=dict(size=7))
st.plotly_chart(fig_2d, use_container_width=True)

# 7. 3차원 산점도
st.markdown("---")
st.subheader("🧊 3차원 산점도")

if len(selected_feature_keys) >= 3:
    col_3d_x, col_3d_y, col_3d_z = st.columns(3)
    
    with col_3d_x:
        x_axis_3d = st.selectbox(
            "3D X축 속성",
            options=selected_feature_keys,
            index=0,
            format_func=lambda x: feature_options[x],
            key="3d_x"
        )
    with col_3d_y:
        y_axis_3d = st.selectbox(
            "3D Y축 속성",
            options=selected_feature_keys,
            index=1,
            format_func=lambda x: feature_options[x],
            key="3d_y"
        )
    with col_3d_z:
        z_axis_3d = st.selectbox(
            "3D Z축 속성",
            options=selected_feature_keys,
            index=2,
            format_func=lambda x: feature_options[x],
            key="3d_z"
        )
        
    fig_3d = px.scatter_3d(
        df,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color='cluster',
        category_orders={'cluster': cluster_order},
        hover_name='movieNm',
        hover_data={'total_audi': ':,', 'first_scrn': ':,', 'days_in_top10': True, 'long_run_index': ':.2f'},
        title=f"3차원 산점도 ({feature_options[x_axis_3d]} x {feature_options[y_axis_3d]} x {feature_options[z_axis_3d]})",
        labels={
            x_axis_3d: feature_options[x_axis_3d],
            y_axis_3d: feature_options[y_axis_3d],
            z_axis_3d: feature_options[z_axis_3d],
            'cluster': '영화 묶음'
        },
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    # 점을 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("💡 3차원 산점도를 그리려면 속성을 3개 이상 선택해야 합니다. 위에서 속성을 추가로 선택해 주세요.")

# 8. 묶음별 요약 통계 표 (원래 단위 평균)
st.markdown("---")
st.subheader("📋 묶음별 영화 편수 및 4대 속성 평균 (원래 단위)")

summary_stats = df.groupby('cluster', observed=False).agg(
    영화편수=('movieNm', 'count'),
    스크린수_평균=('first_scrn', 'mean'),
    누적관객_평균=('total_audi', 'mean'),
    top10일수_평균=('days_in_top10', 'mean'),
    롱런지수_평균=('long_run_index', 'mean')
).reset_index()

summary_stats.columns = ['묶음', '영화 편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']

# 표 포맷팅
summary_formatted = summary_stats.copy()
summary_formatted['영화 편수'] = summary_formatted['영화 편수'].map('{:,}편'.format)
summary_formatted['스크린 수 평균'] = summary_formatted['스크린 수 평균'].map('{:,.1f}개'.format)
summary_formatted['누적 관객 평균'] = summary_formatted['누적 관객 평균'].map('{:,.0f}명'.format)
summary_formatted['10위권 일수 평균'] = summary_formatted['10위권 일수 평균'].map('{:.1f}일'.format)
summary_formatted['롱런 지수 평균'] = summary_formatted['롱런 지수 평균'].map('{:.2f}'.format)

st.dataframe(summary_formatted, use_container_width=True)

# 9. 묶음별 누적 관객수 상위 5개 영화
st.markdown("---")
st.subheader("🏆 묶음별 누적 관객수 대표 영화 Top 5")

col_a, col_b, col_c = st.columns(3)
cols = [col_a, col_b, col_c]

for idx, cluster_name in enumerate(cluster_order):
    cluster_movies = df[df['cluster'] == cluster_name].sort_values(by='total_audi', ascending=False).head(5)
    
    with cols[idx]:
        st.markdown(f"#### 묶음 **{cluster_name}** Top 5")
        for rank, (_, row) in enumerate(cluster_movies.iterrows(), 1):
            st.write(f"**{rank}. {row['movieNm']}**")
            st.caption(f"관객수: {int(row['total_audi']):,}명 | 스크린: {int(row['first_scrn']):,}개")
