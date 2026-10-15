

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 바탕으로 연평균기온의 변화 추세를 분석하고, "
    "선택한 연도의 예상 기온을 확인합니다."
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
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    return df


try:
    daily_data = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다. 데이터 주소와 연결을 확인해 주세요.")
    st.exception(e)
    st.stop()


# ----------------------------------------
# 연도별 평균기온 계산
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

annual_data = annual_data[
    annual_data["관측일수"] >= MIN_OBSERVATION_DAYS
].copy()

annual_data = annual_data.sort_values("연도").reset_index(drop=True)

if len(annual_data) < 2:
    st.error("회귀 분석을 하기에 유효한 연도별 데이터가 부족합니다.")
    st.stop()


# ----------------------------------------
# 전체 기간 회귀 분석
# ----------------------------------------
years = annual_data["연도"].to_numpy(dtype=float)
temperatures = annual_data["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(years, temperatures, 1)
correlation = np.corrcoef(years, temperatures)[0, 1]

start_year = int(annual_data["연도"].min())
end_year = int(annual_data["연도"].max())
year_count = len(annual_data)

# 연간 기울기를 100년당 변화량으로 변환
warming_100_years = slope * 100


# ----------------------------------------
# 최근 20년 회귀 분석
# ----------------------------------------
recent_start_year = end_year - 19

recent_data = annual_data[
    annual_data["연도"] >= recent_start_year
].copy()

if len(recent_data) >= 2:
    recent_years = recent_data["연도"].to_numpy(dtype=float)
    recent_temps = recent_data["연평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(
        recent_years,
        recent_temps,
        1
    )

    recent_warming_100_years = recent_slope * 100
    recent_correlation = np.corrcoef(
        recent_years,
        recent_temps
    )[0, 1]

    recent_actual_start = int(recent_data["연도"].min())
    recent_actual_end = int(recent_data["연도"].max())
else:
    recent_slope = None
    recent_intercept = None
    recent_warming_100_years = None
    recent_correlation = None
    recent_actual_start = None
    recent_actual_end = None


# ----------------------------------------
# 100년당 기온 변화량 크게 표시
# ----------------------------------------
st.subheader("🌡️ 100년당 연평균기온 변화")

st.metric(
    label=f"전체 기간 ({start_year}~{end_year})",
    value=f"{warming_100_years:+.2f} °C",
    help="전체 분석 기간의 선형 회귀 기울기를 100배 한 값입니다."
)

st.caption(
    "양수는 기온 상승 추세, 음수는 기온 하강 추세를 의미합니다. "
    "이는 관측 자료의 선형 추세를 100년 기준으로 환산한 값이며, "
    "실제 미래의 기온 변화가 반드시 이와 같다는 뜻은 아닙니다."
)


# ----------------------------------------
# 전체 기간과 최근 20년 비교
# ----------------------------------------
st.subheader("📊 전체 기간과 최근 20년의 변화 비교")

compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.markdown("### 전체 분석 기간")
    st.metric(
        label=f"{start_year}~{end_year}",
        value=f"{warming_100_years:+.2f} °C / 100년"
    )
    st.write(f"- 분석 연도 수: **{year_count}년**")
    st.write(f"- 연도와 기온의 상관계수: **{correlation:.3f}**")

with compare_col2:
    st.markdown("### 최근 20년")
    if recent_warming_100_years is not None:
        st.metric(
            label=f"{recent_actual_start}~{recent_actual_end}",
            value=f"{recent_warming_100_years:+.2f} °C / 100년",
            delta=(
                f"{recent_warming_100_years - warming_100_years:+.2f} °C"
                " / 100년 (전체 기간 대비)"
            )
        )
        st.write(f"- 분석 연도 수: **{len(recent_data)}년**")
        st.write(f"- 연도와 기온의 상관계수: **{recent_correlation:.3f}**")
    else:
        st.warning("최근 기간의 유효한 연도가 2개 미만이라 비교할 수 없습니다.")

st.caption(
    "최근 20년은 전체 데이터의 마지막 연도를 기준으로 역산합니다. "
    "유효한 관측 자료가 있는 연도만 회귀 계산에 포함합니다. "
    "최근 20년의 기울기도 짧은 기간의 변동 영향을 받을 수 있으므로 "
    "전체 기간의 기울기와 함께 해석하세요."
)


# ----------------------------------------
# 예측할 연도 선택
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
    "전체 분석 기간의 회귀식을 이용한 추정치입니다. "
    "특히 관측 기간에서 멀리 떨어진 연도는 불확실성이 큽니다."
)


# ----------------------------------------
# 산점도와 회귀 직선
# ----------------------------------------
st.subheader("📈 연도별 연평균기온과 회귀 직선")

fig = go.Figure()

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

line_years = np.linspace(start_year, end_year, 200)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

if recent_slope is not None:
    recent_line_years = np.linspace(
        recent_actual_start,
        recent_actual_end,
        100
    )
    recent_line_temps = (
        recent_slope * recent_line_years + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(dash="dash"),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "최근 기간 회귀 추정 기온: %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

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
    title="서울 연평균기온 산점도와 회귀 직선 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------
# 분석에 사용한 연도 정보
# ----------------------------------------
st.info(
    f"전체 회귀 직선은 **{year_count}개 연도**의 자료를 사용했습니다. "
    f"시작 연도는 **{start_year}년**, 끝 연도는 **{end_year}년**입니다. "
    f"2025년 이후 데이터와 관측일이 300일 미만인 해는 제외했습니다."
)


# ----------------------------------------
# 연도별 데이터 표
# ----------------------------------------
with st.expander("연도별 분석 데이터 보기"):
    display_data = annual_data.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)

    display_data = display_data.rename(
        columns={
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

