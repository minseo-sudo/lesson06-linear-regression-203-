st.subheader("연도별 평균기온과 회귀 직선 비교")

try:
    fig = go.Figure()

    # 실제 관측 데이터
    fig.add_trace(
        go.Scatter(
            x=yearly["연도"].tolist(),
            y=yearly["연평균기온"].tolist(),
            mode="markers",
            name="관측 연평균기온",
            marker=dict(
                size=7,
                color="royalblue",
                opacity=0.8,
            ),
            hovertemplate="%{x}년<br>평균기온: %{y:.2f} °C<extra></extra>",
        )
    )

    # 전체 기간 회귀선
    full_x = np.arange(1900, 2101, dtype=float)
    full_y = full_model["slope"] * full_x + full_model["intercept"]

    fig.add_trace(
        go.Scatter(
            x=full_x.tolist(),
            y=full_y.tolist(),
            mode="lines",
            name="전체 기간 회귀선",
            line=dict(color="red", width=3),
        )
    )

    # 최근 20년 회귀선
    recent_x = np.linspace(
        recent_model["start"],
        recent_model["end"],
        100,
    )
    recent_y = (
        recent_model["slope"] * recent_x
        + recent_model["intercept"]
    )

    fig.add_trace(
        go.Scatter(
            x=recent_x.tolist(),
            y=recent_y.tolist(),
            mode="lines",
            name="최근 20년 회귀선",
            line=dict(color="green", width=3, dash="dash"),
        )
    )

    # 선택한 연도의 예측값
    fig.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[predicted_temp],
            mode="markers",
            name=f"{selected_year}년 예상 기온",
            marker=dict(
                size=14,
                color="orange",
                symbol="diamond",
            ),
        )
    )

    fig.update_layout(
        title="서울 연평균기온과 선형회귀 분석",
        template="plotly_white",
        height=600,
        autosize=True,
        xaxis=dict(
            title="연도",
            range=[1900, 2100],
            dtick=20,
        ),
        yaxis=dict(
            title="연평균기온 (°C)",
            autorange=True,
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=50, r=30, t=100, b=60),
    )

    # Streamlit에 그래프 표시
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="temperature_trend_chart",
    )

except Exception as e:
    st.error(f"그래프를 그리는 중 오류가 발생했습니다: {e}")
