import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------
# 기본 설정
# ----------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 분석해 연평균기온의 변화 추세를 살펴보고, "
    "선택한 연도의 예상 기온을 확인해 보세요."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

LAST_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


# ----------------------------------------
# 데이터 불러오기
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜",
        "지점",
        "평균기온",
        "최저기온",
        "최고기온"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    # 날짜와 기온 데이터 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제외
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


try:
    daily_data = load_data()
except Exception as e:
    st.error(
        "기온 데이터를 불러오는 중 문제가 발생했습니다. "
        "데이터 주소와 인터넷 연결을 확인해 주세요."
    )
    st.exception(e)
    st.stop()


# ----------------------------------------
# 연도별 평균기온과 관측일 계산
# ----------------------------------------
annual_data = (
    daily_data[daily_data["연도"] <= LAST_YEAR]
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 이상인 해만 사용
annual_data = annual_data[
    annual_data["관측일수"] >= MIN_OBSERVATION_DAYS
].copy()

annual_data = annual_data.sort_values("연도").reset_index(drop=True)

if len(annual_data) < 2:
    st.error(
        "회귀 직선을 계산하기 위한 데이터가 부족합니다. "
        "데이터의 날짜와 평균기온 열을 확인해 주세요."
    )
    st.stop()


# ----------------------------------------
# 선형 회귀와 상관계수 계산
# ----------------------------------------
years = annual_data["연도"].to_numpy(dtype=float)
temperatures = annual_data["연평균기온"].to_numpy(dtype=float)

# y = 기울기 * 연도 + 절편
slope, intercept = np.polyfit(years, temperatures, 1)

# 피어슨 상관계수
correlation = np.corrcoef(years, temperatures)[0, 1]

start_year = int(annual_data["연도"].min())
end_year = int(annual_data["연도"].max())
year_count = len(annual_data)


# ----------------------------------------
# 연도 선택 슬라이더
# ----------------------------------------
st.subheader("📅 예측할 연도 선택")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(
    "위 값은 선택한 연도에 회귀 직선을 적용해 계산한 추정치입니다. "
    "특히 관측 기간에서 멀리 떨어진 연도의 값은 불확실성이 큽니다."
)


# ----------------------------------------
# 분석 데이터 정보
# ----------------------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("회귀 분석에 사용한 연도 수", f"{year_count:,}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

st.write(
    f"**상관계수:** {correlation:.4f}"
)

st.caption(
    "상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다. "
    "양수이면 함께 증가하는 경향, 음수이면 한쪽이 증가할 때 "
    "다른 쪽이 감소하는 경향을 뜻합니다. "
    "상관관계만으로 원인과 결과를 확정할 수는 없습니다."
)


# ----------------------------------------
# 산점도와 회귀 직선
# ----------------------------------------
st.subheader("📈 연도별 연평균기온과 회귀 직선")

# 회귀 직선은 실제 분석에 사용한 기간에 표시
line_years = np.linspace(start_year, end_year, 200)
line_temps = slope * line_years + intercept

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual_data["연도"],
        y=annual_data["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        customdata=annual_data[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형 회귀 직선",
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 예상값 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=12, symbol="diamond"),
        hovertemplate=(
            "선택 연도: %{x:.0f}년<br>"
            "예상 연평균기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온 산점도와 선형 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"이 회귀 직선은 **{start_year}년부터 {end_year}년까지**의 "
    f"연평균기온 {year_count}개를 사용해 계산했습니다. "
    f"각 연도는 평균기온 관측일이 300일 이상인 경우에만 포함했습니다. "
    f"2025년 이후의 데이터는 분석에서 제외했습니다."
)


# ----------------------------------------
# 연도별 분석 데이터 확인
# ----------------------------------------
with st.expander("연도별 분석 데이터 보기"):
    display_data = annual_data.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)

    display_data = display_data.rename(
        columns={
            "연도": "연도",
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


