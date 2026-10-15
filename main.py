import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(page_title="기온 예측기", layout="wide")
st.title("기온 예측기")


@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 2025년까지, 유효 관측일 300일 이상인 해만 사용
    df = df[df["연도"] <= 2025]
    grouped = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )
    valid = grouped["관측일수"] >= 300
    return grouped[valid].sort_values("연도").reset_index(drop=True)


def fit_line(data):
    """연도별 평균기온에 직선을 맞춘다."""
    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    if len(x) < 2 or len(np.unique(x)) < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


yearly = load_yearly()

if len(yearly) < 2:
    st.error("회귀 분석에 사용할 연도 데이터가 부족합니다.")
    st.stop()

# 전체 기간 회귀
a, b = fit_line(yearly)

# 데이터에서 가장 최근 연도를 기준으로 최근 20년 구간 설정
latest_year = int(yearly["연도"].max())
recent_start = latest_year - 19
recent = yearly[
    (yearly["연도"] >= recent_start)
    & (yearly["연도"] <= latest_year)
]

recent_fit = fit_line(recent)

# -----------------------------
# 100년당 기온 변화량 비교
# -----------------------------
st.subheader("100년당 기온 변화 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        f"전체 기간 ({int(yearly['연도'].min())}~{latest_year}년)",
        f"{a * 100:+.2f} ℃ / 100년",
    )

with col2:
    if recent_fit is not None:
        recent_a, recent_b = recent_fit
        st.metric(
            f"최근 20년 ({recent_start}~{latest_year}년)",
            f"{recent_a * 100:+.2f} ℃ / 100년",
            delta=f"{(recent_a - a) * 100:+.2f} ℃ / 100년 (전체 대비)",
        )
    else:
        st.warning("최근 20년 데이터가 부족해 기울기를 계산할 수 없습니다.")

st.caption(
    "기울기에 100을 곱한 값입니다. 양수는 상승, 음수는 하락 경향을 뜻합니다. "
    "최근 20년의 변화 속도가 앞으로 100년간 그대로 유지된다는 뜻은 아닙니다."
)

# -----------------------------
# 산점도 및 회귀 직선
# -----------------------------
fig = px.scatter(
    yearly,
    x="연도",
    y="연평균기온",
    opacity=0.6,
    hover_data=["관측일수"],
    labels={"연도": "연도", "연평균기온": "연평균기온 (℃)"},
)

# 전체 기간 회귀 직선
line_years = np.arange(1900, 2101)
fig.add_scatter(
    x=line_years,
    y=a * line_years + b,
    mode="lines",
    name=f"전체 기간 ({a * 100:+.2f} ℃/100년)",
    line=dict(color="red", width=3),
)

# 최근 20년 회귀 직선은 최근 구간에서만 표시
if recent_fit is not None:
    recent_a, recent_b = recent_fit
    recent_years = np.linspace(
        recent["연도"].min(),
        recent["연도"].max(),
        100,
    )
    fig.add_scatter(
        x=recent_years,
        y=recent_a * recent_years + recent_b,
        mode="lines",
        name=f"최근 20년 ({recent_a * 100:+.2f} ℃/100년)",
        line=dict(color="green", width=3, dash="dash"),
    )

fig.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"직선을 만든 해: {len(yearly)}개 "
    f"({int(yearly['연도'].min())}~{latest_year}년)"
)

# 상관계수
r = yearly["연도"].corr(yearly["연평균기온"])
st.metric("연도와 연평균기온의 상관계수", f"{r:.3f}")

# -----------------------------
# 선택 연도 예측
# -----------------------------
year = st.slider("예측할 연도를 고르세요", 1900, 2100, 2045)
st.metric(f"{year}년 예상 연평균기온", f"{a * year + b:.1f} ℃")

if year < yearly["연도"].min() or year > yearly["연도"].max():
    st.info("학습 범위 밖의 외삽값입니다. 실제 미래 기온을 보장하지 않습니다.")
