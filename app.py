
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime


# =========================================================
# 기본 설정
# =========================================================

TICKER = "ATCH"

st.set_page_config(
    page_title="ATCH AI Market Analyzer",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# 제목
# =========================================================

st.title("📈 ATCH AI Market Analyzer")

st.caption(
    "AtlasClear Holdings (ATCH) "
    "주가 분석 및 LSTM 예측 프로그램"
)


# =========================================================
# 데이터 갱신 버튼
# =========================================================

left, right = st.columns([1, 3])

with left:

    if st.button(
        "🔄 최신 데이터 갱신",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

with right:

    st.info(
        "버튼을 누르면 저장된 데이터를 삭제하고 "
        "최신 데이터를 다시 조회합니다."
    )


st.caption(
    "조회 시간: "
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)


# =========================================================
# 사이드바
# =========================================================

st.sidebar.header("⚙️ 분석 설정")

period = st.sidebar.selectbox(
    "주가 조회 기간",
    ["1y", "2y", "5y", "max"],
    index=1
)

lookback = st.sidebar.slider(
    "LSTM 입력 기간",
    20,
    120,
    60,
    10
)

epochs = st.sidebar.slider(
    "LSTM 학습 횟수",
    5,
    30,
    10
)

future_days = st.sidebar.slider(
    "미래 예측 거래일",
    1,
    5,
    5
)


# =========================================================
# 최신 가격
# =========================================================

@st.cache_data(ttl=30)
def get_current_price():

    try:

        ticker = yf.Ticker(TICKER)

        info = ticker.fast_info

        price = info.get(
            "last_price",
            np.nan
        )

        previous = info.get(
            "previous_close",
            np.nan
        )

        volume = info.get(
            "last_volume",
            np.nan
        )

        return price, previous, volume

    except Exception:

        return np.nan, np.nan, np.nan


price, previous, live_volume = get_current_price()


if (
    np.isfinite(price)
    and np.isfinite(previous)
    and previous != 0
):

    change = (
        (price - previous)
        / previous
        * 100
    )

else:

    change = np.nan


# =========================================================
# 가격 카드
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "현재/최근 가격",
        f"${price:.4f}"
        if np.isfinite(price)
        else "N/A",
        f"{change:+.2f}%"
        if np.isfinite(change)
        else None
    )

with c2:

    st.metric(
        "전일 종가",
        f"${previous:.4f}"
        if np.isfinite(previous)
        else "N/A"
    )

with c3:

    st.metric(
        "최근 거래량",
        f"{live_volume:,.0f}"
        if np.isfinite(live_volume)
        else "N/A"
    )

with c4:

    st.metric(
        "종목",
        TICKER
    )


# =========================================================
# 과거 주가
# =========================================================

@st.cache_data(ttl=300)
def get_history(period):

    data = yf.download(
        TICKER,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:

        return data

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data = data.reset_index()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.dropna(
        subset=["Close"]
    )

    return data


df = get_history(period)


if df.empty:

    st.error(
        "주가 데이터를 가져오지 못했습니다."
    )

    st.stop()


# =========================================================
# 기술적 지표
# =========================================================

df["MA20"] = (
    df["Close"]
    .rolling(20)
    .mean()
)

df["MA60"] = (
    df["Close"]
    .rolling(60)
    .mean()
)

df["Return"] = (
    df["Close"]
    .pct_change()
)

df["Volatility20"] = (
    df["Return"]
    .rolling(20)
    .std()
)

df["VolumeMA20"] = (
    df["Volume"]
    .rolling(20)
    .mean()
)


# =========================================================
# 주가 차트
# =========================================================

st.header("① 주가 차트")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="ATCH"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA20"],
        name="20일 이동평균"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA60"],
        name="60일 이동평균"
    )
)

fig.update_layout(
    height=600,
    title="ATCH 주가 및 이동평균"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 거래량
# =========================================================

st.header("② 거래량")

volume_df = df.tail(120)

volume_fig = go.Figure()

volume_fig.add_trace(
    go.Bar(
        x=volume_df["Date"],
        y=volume_df["Volume"],
        name="거래량"
    )
)

volume_fig.add_trace(
    go.Scatter(
        x=volume_df["Date"],
        y=volume_df["VolumeMA20"],
        name="20일 평균 거래량"
    )
)

volume_fig.update_layout(
    height=400
)

st.plotly_chart(
    volume_fig,
    use_container_width=True
)


# =========================================================
# 변동성
# =========================================================

st.header("③ 변동성")

vol_fig = px.line(
    df,
    x="Date",
    y="Volatility20",
    title="20일 변동성"
)

st.plotly_chart(
    vol_fig,
    use_container_width=True
)


# =========================================================
# LSTM
# =========================================================

st.header("④ LSTM 인공신경망")


close = (
    df["Close"]
    .astype(float)
    .values
    .reshape(-1, 1)
)


if len(close) <= lookback + 30:

    st.error(
        "LSTM 학습에 필요한 데이터가 부족합니다."
    )

    st.stop()


split = int(
    len(close) * 0.8
)


scaler = MinMaxScaler()

scaler.fit(
    close[:split]
)

scaled = scaler.transform(
    close
)


X = []
y = []


for i in range(
    lookback,
    len(scaled)
):

    X.append(
        scaled[
            i-lookback:i,
            0
        ]
    )

    y.append(
        scaled[i, 0]
    )


X = np.array(X)

y = np.array(y)


X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)


train_end = (
    split - lookback
)


X_train = X[:train_end]

y_train = y[:train_end]

X_test = X[train_end:]

y_test = y[train_end:]


# =========================================================
# 모델
# =========================================================

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            lookback,
            1
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(32)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(
    Dense(1)
)


model.compile(
    optimizer="adam",
    loss="mse"
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# =========================================================
# 학습
# =========================================================

with st.spinner(
    "LSTM 모델을 학습하고 있습니다..."
):

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )


# =========================================================
# 평가
# =========================================================

pred_scaled = model.predict(
    X_test,
    verbose=0
)


pred = scaler.inverse_transform(
    pred_scaled
).flatten()


actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()


mae = mean_absolute_error(
    actual,
    pred
)


rmse = np.sqrt(
    mean_squared_error(
        actual,
        pred
    )
)


m1, m2 = st.columns(2)

with m1:

    st.metric(
        "MAE",
        f"${mae:.4f}"
    )

with m2:

    st.metric(
        "RMSE",
        f"${rmse:.4f}"
    )


# =========================================================
# 실제값과 예측값
# =========================================================

st.subheader(
    "실제 가격 vs LSTM 예측"
)


test_dates = (
    df["Date"]
    .iloc[split:]
    .reset_index(drop=True)
)


compare = go.Figure()


compare.add_trace(
    go.Scatter(
        x=test_dates,
        y=actual,
        name="실제 가격"
    )
)

compare.add_trace(
    go.Scatter(
        x=test_dates,
        y=pred,
        name="LSTM 예측"
    )
)


compare.update_layout(
    height=500
)


st.plotly_chart(
    compare,
    use_container_width=True
)


# =========================================================
# 학습 손실
# =========================================================

st.subheader(
    "모델 학습 과정"
)


loss_fig = go.Figure()


loss_fig.add_trace(
    go.Scatter(
        y=history.history["loss"],
        name="Training Loss"
    )
)

loss_fig.add_trace(
    go.Scatter(
        y=history.history["val_loss"],
        name="Validation Loss"
    )
)


loss_fig.update_layout(
    height=350
)


st.plotly_chart(
    loss_fig,
    use_container_width=True
)


# =========================================================
# 미래 예측
# =========================================================

st.header("⑤ 미래 주가 예측")


sequence = (
    scaled[-lookback:]
    .reshape(
        1,
        lookback,
        1
    )
)


future_scaled = []


for _ in range(
    future_days
):

    next_value = model.predict(
        sequence,
        verbose=0
    )[0][0]

    future_scaled.append(
        next_value
    )

    next_array = np.array(
        [[[next_value]]]
    )

    sequence = np.concatenate(
        [
            sequence[:, 1:, :],
            next_array
        ],
        axis=1
    )


future_prices = scaler.inverse_transform(
    np.array(
        future_scaled
    ).reshape(-1, 1)
).flatten()


future_df = pd.DataFrame(
    {
        "예측 거래일":
            range(
                1,
                future_days + 1
            ),

        "예측 종가":
            future_prices
    }
)


st.dataframe(
    future_df,
    use_container_width=True
)


future_fig = px.line(
    future_df,
    x="예측 거래일",
    y="예측 종가",
    markers=True,
    title="미래 주가 예측"
)


st.plotly_chart(
    future_fig,
    use_container_width=True
)


# =========================================================
# 실적
# =========================================================

st.header("⑥ 실적 발표")


@st.cache_data(ttl=3600)
def get_earnings():

    try:

        ticker = yf.Ticker(TICKER)

        data = ticker.get_earnings_dates(
            limit=12
        )

        if data is None:

            return pd.DataFrame()

        return data.reset_index()

    except Exception:

        return pd.DataFrame()


earnings = get_earnings()


if not earnings.empty:

    st.dataframe(
        earnings,
        use_container_width=True
    )

else:

    st.info(
        "실적 발표 데이터를 제공받지 못했습니다."
    )


# =========================================================
# SEC
# =========================================================

st.header("⑦ SEC 공시")


SEC_URL = (
    "https://data.sec.gov/"
    "submissions/"
    "CIK0001963088.json"
)


@st.cache_data(ttl=1800)
def get_sec():

    headers = {
        "User-Agent":
        "ATCH educational project contact@example.com"
    }

    try:

        response = requests.get(
            SEC_URL,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        recent = data[
            "filings"
        ][
            "recent"
        ]

        result = pd.DataFrame(
            recent
        )

        return result

    except Exception:

        return pd.DataFrame()


sec = get_sec()


if not sec.empty:

    forms = [
        "10-K",
        "10-Q",
        "8-K",
        "3",
        "4",
        "5"
    ]

    important = sec[
        sec["form"].isin(forms)
    ].head(30)


    columns = [
        "filingDate",
        "form",
        "accessionNumber",
        "primaryDocument"
    ]


    columns = [
        c
        for c in columns
        if c in important.columns
    ]


    st.dataframe(
        important[columns],
        use_container_width=True
    )

else:

    st.info(
        "SEC 공시 데이터를 가져오지 못했습니다."
    )


# =========================================================
# 공매도
# =========================================================

st.header("⑧ 공매도 정보")


st.info(
    "공매도 데이터는 주가와 달리 "
    "초단위로 갱신되는 데이터가 아닐 수 있습니다."
)


st.link_button(
    "공매도 정보 확인",
    "https://chartexchange.com/"
    "symbol/nyseamerican-atch/"
    "short-interest/"
)


st.write(
    "확인 가능한 주요 지표:"
)

st.write(
    "- Short Interest"
)

st.write(
    "- Days to Cover"
)

st.write(
    "- Short Volume"
)

st.write(
    "- Borrow Fee"
)


# =========================================================
# 요약
# =========================================================

st.header("⑨ 시장 데이터 요약")


latest = df.iloc[-1]


a, b, c = st.columns(3)


with a:

    st.metric(
        "최근 종가",
        f"${float(latest['Close']):.4f}"
    )


with b:

    value = latest["Volatility20"]

    st.metric(
        "20일 변동성",
        f"{value:.4f}"
        if pd.notna(value)
        else "N/A"
    )


with c:

    value = latest["VolumeMA20"]

    st.metric(
        "20일 평균 거래량",
        f"{value:,.0f}"
        if pd.notna(value)
        else "N/A"
    )


# =========================================================
# 알고리즘 설명
# =========================================================

st.header("⑩ 알고리즘")


st.write(
    "1. 공개 주가 데이터를 수집합니다."
)

st.write(
    "2. 종가 데이터를 0~1 범위로 정규화합니다."
)

st.write(
    "3. 최근 거래일 데이터를 LSTM 입력값으로 구성합니다."
)

st.write(
    "4. LSTM 인공신경망을 학습합니다."
)

st.write(
    "5. 테스트 데이터에서 MAE와 RMSE를 계산합니다."
)

st.write(
    "6. 학습된 모델을 이용하여 미래 거래일 가격을 예측합니다."
)

st.write(
    "7. 실적, SEC 공시, 공매도 관련 정보를 함께 확인합니다."
)


# =========================================================
# 주의사항
# =========================================================

st.warning(
    "본 프로그램은 교육 및 데이터 분석 목적입니다. "
    "인공지능 예측값은 실제 미래 주가를 보장하지 않습니다."
)
