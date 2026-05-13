import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 페이지 설정
st.set_page_config(
    page_title="비트코인 AI 예측 대시보드",
    page_icon="₿",
    layout="wide"
)

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    # 세미콜론(;) 구분자로 CSV 읽기
    df = pd.read_csv(file_path, sep=';')
    
    # 시간 데이터 변환
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    df = df.sort_values('timeOpen')
    
    # 기술적 지표
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    return df

# 선형 회귀를 이용한 내일 가격 예측 함수
def predict_next_day(df):
    # 최근 30일 데이터를 학습용으로 사용
    window_size = 30
    if len(df) < window_size:
        window_size = len(df)
    
    recent_data = df.tail(window_size).copy()
    
    # X값: 날짜를 숫자로 변환 (0, 1, 2...), y값: 종가
    X = np.array(range(len(recent_data))).reshape(-1, 1)
    y = recent_data['close'].values
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 다음 날(현재 데이터의 마지막 날 + 1) 예측
    next_day_index = np.array([[len(recent_data)]])
    prediction = model.predict(next_day_index)[0]
    
    return prediction

FILE_NAME = 'coin.csv'

try:
    df = load_data(FILE_NAME)

    if df is None:
        st.error(f"'{FILE_NAME}' 파일을 찾을 수 없습니다. 같은 폴더에 파일을 넣어주세요.")
        st.stop()

    # 사이드바 설정
    st.sidebar.header("📊 분석 및 예측 설정")
    st.sidebar.markdown("---")
    
    # 날짜 범위 선택
    min_date = df['timeOpen'].min().date()
    max_date = df['timeOpen'].max().date()
    date_range = st.sidebar.date_input("분석 기간", value=(min_date, max_date))

    # 데이터 필터링
    if len(date_range) == 2:
        mask = (df['timeOpen'].dt.date >= date_range[0]) & (df['timeOpen'].dt.date <= date_range[1])
        filtered_df = df.loc[mask].copy()
    else:
        filtered_df = df.copy()

    # 상단 레이아웃: 제목 및 AI 예측 결과
    st.title("₿ 비트코인(BTC) AI 가격 예측 대시보드")
    
    # 예측 세션
    st.subheader("🚀 내일 가격 예측 (AI 선형회귀 모델)")
    
    # 전체 데이터를 기반으로 예측 (가장 최신 시점 기준)
    predicted_price = predict_next_day(df)
    current_price = df['close'].iloc[-1]
    last_date = df['timeOpen'].iloc[-1]
    next_date = last_date + timedelta(days=1)
    
    change = predicted_price - current_price
    change_percent = (change / current_price) * 100
    
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    
    with p_col1:
        st.metric("오늘 현재가", f"₩{current_price:,.0f}")
    with p_col2:
        color = "normal" if change > 0 else "inverse"
        st.metric(f"내일 예측가 ({next_date.strftime('%m/%d')})", 
                  f"₩{predicted_price:,.0f}", 
                  f"{change_percent:+.2f}%", 
                  delta_color=color)
    
    with p_col3:
        if change > 0:
            st.success(f"📈 **상승 예측:** 내일 가격이 약 **{change:,.0f}원** 상승할 것으로 분석되었습니다.")
        else:
            st.error(f"📉 **하락 예측:** 내일 가격이 약 **{abs(change):,.0f}원** 하락할 것으로 분석되었습니다.")
            
    st.markdown("---")

    # 주요 지표 (선택 기간 기준)
    if not filtered_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("최근 종가", f"₩{filtered_df['close'].iloc[-1]:,.0f}")
        col2.metric("기간 내 최고가", f"₩{filtered_df['high'].max():,.0f}")
        col3.metric("기간 내 최저가", f"₩{filtered_df['low'].min():,.0f}")
        col4.metric("평균 거래량", f"{filtered_df['volume'].mean():,.0f}")

        # 메인 시각화 차트
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            subplot_titles=('가격 추이 및 이동평균선', '거래량'), 
                            row_width=[0.3, 0.7])

        # 캔들스틱
        fig.add_trace(go.Candlestick(
            x=filtered_df['timeOpen'], open=filtered_df['open'],
            high=filtered_df['high'], low=filtered_df['low'], close=filtered_df['close'],
            name='BTC'
        ), row=1, col=1)

        # 예측 지점 표시
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[current_price, predicted_price],
            mode='lines+markers',
            line=dict(color='yellow', dash='dash'),
            name='AI 예측 경로'
        ), row=1, col=1)

        # 거래량
        colors = ['red' if r['open'] > r['close'] else 'green' for _, r in filtered_df.iterrows()]
        fig.add_trace(go.Bar(x=filtered_df['timeOpen'], y=filtered_df['volume'], 
                             marker_color=colors, name='거래량'), row=2, col=1)

        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 상세 데이터
        with st.expander("원본 데이터 확인"):
            st.write(filtered_df.sort_values('timeOpen', ascending=False))

    else:
        st.warning("선택한 날짜 범위에 데이터가 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
