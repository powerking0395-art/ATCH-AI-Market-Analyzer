from pathlib import Path
import zipfile

project = Path("/content/ATCH-AI-Market-Analyzer")
project.mkdir(exist_ok=True)

# ---------------------------------------------------------
# app.py
# ---------------------------------------------------------

app_code = r'''import streamlit as st
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


TICKER = "ATCH"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="ATCH AI Market Analyzer",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# REFRESH
# =========================================================

c1, c2 = st.columns([1, 4])

with c1:

    if st.button(
        "🔄 최신 데이터 갱신",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


with c2:

    st.caption(
        "버튼을 누르면 저장된 데이터를 삭제하고 최신 데이터를 다시 조회합니다."
    )


# =========================================================
# TITLE
# =========================================================

st.title("📈 ATCH AI Market Analyzer")

st.caption(
    "AtlasClear Holdings (ATCH) | "
    "주가 · 거래량 · LSTM · 실적 · SEC 공시 · 공매도"
)


st.info(
    "데이터 제공처의 갱신 주기에 따라 실제 거래소의 "
    "초단위 실시간 가격과 차이가 있을 수 있습니다."
)


st.caption(
    "페이지 조회 시간: "
    + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ 분석 설정")


period = st.sidebar.selectbox(
    "주가 데이터 기간",
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
    "학습 Epoch",
    5,
    50,
    20
)


future_days = st.sidebar.slider(
    "미래 예측 거래일",
    1,
    5,
    5
)


# =========================================================
# LIVE PRICE
# =========================================================

@st.cache_data(ttl=30)
def get_live_price():

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


price, previous, live_volume = get_live_price()


if (
    np.isfinite(price)
    and np.isfinite(previous)
    and previous != 0
):

    change_pct = (
        (price - previous)
        / previous
        * 100
    )

else:

    change_pct = np.nan


# =========================================================
# PRICE CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "현재/최근 가격",
        f"${price:.4f}"
        if np.isfinite(price)
        else "N/A",
        f"{change_pct:+.2f}%"
        if np.isfinite(change_pct)
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
# HISTORY
# =========================================================

@st.cache_data(ttl=300)
def get_history(selected_period):

    data = yf.download(
        TICKER,
        period=selected_period,
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
# INDICATORS
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
# PRICE CHART
# =========================================================

st.header("① 주가")


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
    title="ATCH 주가",
    height=600,
    xaxis_rangeslider_visible=True
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# VOLUME
# =========================================================

st.header("② 거래량")


volume_data = df.tail(120)


volume_fig = go.Figure()


volume_fig.add_trace(
    go.Bar(
        x=volume_data["Date"],
        y=volume_data["Volume"],
        name="거래량"
    )
)


volume_fig.add_trace(
    go.Scatter(
        x=volume_data["Date"],
        y=volume_data["VolumeMA20"],
        name="20일 평균"
    )
)


volume_fig.update_layout(
    height=400,
    title="최근 120거래일 거래량"
)


st.plotly_chart(
    volume_fig,
    use_container_width=True
)


# =========================================================
# VOLATILITY
# =========================================================

st.header("③ 변동성")


vol_fig = px.line(
    df,
    x="Date",
    y="Volatility20",
    title="20일 이동 변동성"
)


st.plotly_chart(
    vol_fig,
    use_container_width=True
)


# =========================================================
# LSTM DATA
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
        "LSTM 학습 데이터가 부족합니다."
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
            i - lookback:i,
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
# MODEL
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
    patience=5,
    restore_best_weights=True
)


# =========================================================
# TRAIN
# =========================================================

with st.spinner(
    "LSTM 모델 학습 중..."
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
# EVALUATION
# =========================================================

prediction_scaled = model.predict(
    X_test,
    verbose=0
)


prediction = (
    scaler
    .inverse_transform(
        prediction_scaled
    )
    .flatten()
)


actual = (
    scaler
    .inverse_transform(
        y_test.reshape(-1, 1)
    )
    .flatten()
)


mae = mean_absolute_error(
    actual,
    prediction
)


rmse = np.sqrt(
    mean_squared_error(
        actual,
        prediction
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
# ACTUAL VS PREDICTION
# =========================================================

test_dates = (
    df["Date"]
    .iloc[split:]
    .reset_index(drop=True)
)


compare_fig = go.Figure()


compare_fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=actual,
        name="실제 가격"
    )
)


compare_fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=prediction,
        name="LSTM 예측"
    )
)


compare_fig.update_layout(
    title="실제 가격 vs LSTM 예측",
    height=500
)


st.plotly_chart(
    compare_fig,
    use_container_width=True
)


# =========================================================
# TRAINING LOSS
# =========================================================

st.header("⑤ 모델 학습 과정")


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
    title="Training / Validation Loss",
    height=350
)


st.plotly_chart(
    loss_fig,
    use_container_width=True
)


# =========================================================
# FUTURE PREDICTION
# =========================================================

st.header("⑥ 미래 주가 예측")


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

    next_prediction = model.predict(
        sequence,
        verbose=0
    )[0][0]


    future_scaled.append(
        next_prediction
    )


    new_value = np.array(
        [[[next_prediction]]]
    )


    sequence = np.concatenate(
        [
            sequence[:, 1:, :],
            new_value
        ],
        axis=1
    )


future_prices = (
    scaler.inverse_transform(
        np.array(
            future_scaled
        ).reshape(-1, 1)
    )
    .flatten()
)


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
# EARNINGS
# =========================================================

st.header("⑦ 실적 발표")


@st.cache_data(ttl=3600)
def get_earnings():

    try:

        ticker = yf.Ticker(TICKER)

        result = ticker.get_earnings_dates(
            limit=12
        )

        if result is None:

            return pd.DataFrame()


        return result.reset_index()


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
        "실적 발표 데이터를 가져오지 못했습니다."
    )


# =========================================================
# SEC
# =========================================================

st.header("⑧ SEC 공시")


st.write(
    "SEC EDGAR의 공개 공시 데이터를 확인합니다."
)


SEC_URL = (
    "https://data.sec.gov/"
    "submissions/"
    "CIK0001963088.json"
)


@st.cache_data(ttl=1800)
def get_sec():

    headers = {
        "User-Agent":
        "ATCH Educational Project contact@example.com"
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


        filings = pd.DataFrame(
            recent
        )


        filings["filingDate"] = pd.to_datetime(
            filings["filingDate"]
        )


        return filings.sort_values(
            "filingDate",
            ascending=False
        )


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

    st.warning(
        "SEC 데이터를 가져오지 못했습니다."
    )


# =========================================================
# SHORT INTEREST
# =========================================================

st.header("⑨ 공매도 정보")


st.info(
    "공매도 데이터는 주가와 달리 초단위로 "
    "갱신되지 않을 수 있습니다."
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Short Interest"
    )


    st.link_button(
        "공매도 정보 확인",
        "https://chartexchange.com/"
        "symbol/nyseamerican-atch/"
        "short-interest/"
    )


with col2:

    st.subheader(
        "주요 지표"
    )


    st.write(
        "• Short Interest"
    )


    st.write(
        "• Days to Cover"
    )


    st.write(
        "• Short Volume"
    )


    st.write(
        "• Borrow Fee"
    )


    st.write(
        "• Failure to Deliver"
    )


# =========================================================
# SUMMARY
# =========================================================

st.header("⑩ 시장 데이터 요약")


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
# ALGORITHM
# =========================================================

st.header("⑪ 알고리즘 설명")


st.write(
    "① 공개 주가 데이터를 수집합니다."
)


st.write(
    "② 종가 데이터를 Min-Max Scaling으로 정규화합니다."
)


st.write(
    "③ 최근 거래일 데이터를 LSTM 입력 시퀀스로 구성합니다."
)


st.write(
    "④ LSTM 신경망을 학습합니다."
)


st.write(
    "⑤ 테스트 데이터에서 MAE와 RMSE를 계산합니다."
)


st.write(
    "⑥ 학습된 모델을 이용하여 미래 거래일을 예측합니다."
)


st.write(
    "⑦ 실적, SEC 공시, 공매도 관련 정보를 함께 확인합니다."
)


# =========================================================
# WARNING
# =========================================================

st.warning(
    "본 프로그램은 교육 및 데이터 분석 목적입니다. "
    "AI 예측값은 실제 미래 주가를 보장하지 않습니다."
)


st.caption(
    "Data sources: Yahoo Finance / SEC EDGAR / "
    "public short-interest resources"
)
'''


# ---------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------

requirements = """streamlit
yfinance
tensorflow
scikit-learn
pandas
numpy
plotly
requests
"""


# ---------------------------------------------------------
# 파일 저장
# ---------------------------------------------------------

app_path = project / "app.py"

requirements_path = project / "requirements.txt"

app_path.write_text(
    app_code,
    encoding="utf-8"
)

requirements_path.write_text(
    requirements,
    encoding="utf-8"
)


# ---------------------------------------------------------
# 문법 검사
# ---------------------------------------------------------

import ast

with open(
    app_path,
    "r",
    encoding="utf-8"
) as f:

    code = f.read()


ast.parse(code)


# ---------------------------------------------------------
# ZIP 생성
# ---------------------------------------------------------

zip_path = Path(
    "/content/ATCH-AI-Market-Analyzer-GITHUB.zip"
)


with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    z.write(
        app_path,
        "ATCH-AI-Market-Analyzer/app.py"
    )

    z.write(
        requirements_path,
        "ATCH-AI-Market-Analyzer/requirements.txt"
    )


print("=" * 50)
print("✅ 복사본 제작 완료")
print("✅ Python 문법 검사 통과")
print("=" * 50)
print()
print("파일:")
print(app_path)
print(requirements_path)
print()
print("GitHub 업로드용 ZIP:")
print(zip_path)
