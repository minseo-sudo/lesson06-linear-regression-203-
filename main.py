import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
CUTOFF_YEAR = 2025
MIN_DAYS = 300


# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_temperature_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")

    # 날짜와 기온을 숫자형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 유효하지 않은 행 제외
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 수업 기준 연도까지만 사용
    df = df[df["연도"] <= CUTOFF_YEAR].copy()

    # 연도별 평균기온 및 실제 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 사용
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


# -----------------------------
# 앱 화면
# -----------------------------
st.title("🌡️ 서울 기온 예측기")
st.caption(
    "서울의 연평균기온 데이터를 이용한 선형회귀 예측입니다. "
    "예측 결과는 통계적 추정치이며 실제 기후 예측과는 다를 수 있습니다."
)

try:
    yearly = load_temperature_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소를 확인한 뒤 다시 실행해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 수행하기에 유효한 연도 데이터가 부족합니다.")
    st.stop()

# -----------------------------
# 선형회귀 및 상관계수
# -----------------------------
years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# y = slope * year + intercept
slope, intercept = np.polyfit(years, temps, 1)
correlation = float(np.corrcoef(years, temps)[0, 1])

start_year = int(years.min())
end_year = int(years.max())
year_count = len(yearly)

# -----------------------------
# 연도 슬라이더와 예측값
# -----------------------------
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
    format="%d년",
)

predicted_temp = slope * selected_year + intercept

st.subheader(f"{selected_year}년 예상 연평균기온")
st.metric(
    label="선형회귀 예측값",
    value=f"{predicted_temp:.2f} °C",
)

if selected_year > end_year:
    st.caption(
        f"{selected_year}년은 학습 데이터의 마지막 연도({end_year}년) "
        "이후이므로 외삽 결과입니다. 불확실성이 큽니다."
    )
elif selected_year < start_year:
    st.caption(
        f"{selected_year}년은 학습 데이터의 시작 연도({start_year}년) "
        "이전이므로 외삽 결과입니다. 불확실성이 큽니다."
    )

# -----------------------------
# 회귀 분석 정보
# -----------------------------
st.subheader("회귀 분석 정보")

col1, col2, col3 = st.columns(3)
col1.metric("학습에 사용한 연도 수", f"{year_count}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")

st.write(f"**상관계수 (Pearson r):** `{correlation:.4f}`")
st.write(
    f"**회귀식:** 연평균기온 = "
    f"`{slope:.6f} × 연도 + ({intercept:.4f})`"
)
st.caption(
    f"포함 기준: {CUTOFF_YEAR}년까지, "
    f"유효한 관측일 {MIN_DAYS}일 이상인 연도. "
    "관측일 수는 유효한 평균기온이 있는 서로 다른 날짜를 기준으로 계산합니다."
)

# -----------------------------
# Plotly 산점도 + 회귀 직선
# -----------------------------
st.subheader("연도별 평균기온과 회귀 직선")

# 학습 구간에서 직선을 그린 뒤 1900~2100년 예측 구간도 표시
line_years = np.arange(1900, 2101)
line_temps = slope * line_years + intercept

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="관측 연평균기온",
        marker=dict(size=7, color="#2471A3", opacity=0.8),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형회귀 직선",
        line=dict(color="#E74C3C", width=2.5),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 슬라이더로 선택한 연도의 예측 위치
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(
            size=13,
            color="#F39C12",
            symbol="diamond",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 연평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 학습 데이터가 있는 구간 표시
fig.add_vrect(
    x0=start_year,
    x1=end_year,
    fillcolor="LightBlue",
    opacity=0.12,
    line_width=0,
    layer="below",
    annotation_text="학습 데이터 구간",
    annotation_position="top left",
)

fig.update_layout(
    template="plotly_white",
    height=560,
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=20, r=20, t=80, b=20),
)
fig.update_xaxes(range=[1900, 2100], dtick=20)
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 연도별 집계 데이터 확인
# -----------------------------
with st.expander("학습에 사용된 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(
            columns={
                "연도": "연도",
                "연평균기온": "연평균기온 (°C)",
                "관측일수": "관측일수 (일)",
            }
        ).style.format(
            {
                "연평균기온 (°C)": "{:.2f}",
                "관측일수 (일)": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "주의: 연도와 평균기온의 선형 관계만 사용한 단순 회귀 모델입니다. "
    "장기적인 기후 변화, 온실가스 배출, 자연 변동성 등을 별도로 "
    "모델링하지 않으므로 2100년 등의 값은 신뢰도 높은 기후 전망이 아닙니다."
)
