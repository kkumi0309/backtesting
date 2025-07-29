import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_sp500_data():
    """S&P 500 데이터 로드"""
    df = pd.read_excel('sp500_data.xlsx')
    date_column = df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)
    return df

def load_rsi_data():
    """RSI 데이터 로드"""
    df = pd.read_excel('RSI_DATE.xlsx', skiprows=1)
    date_column = df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)
    return df['RSI'].dropna()

def classify_market_season(rsi):
    """RSI 값을 기반으로 시장 계절을 분류"""
    def get_season(rsi_value):
        if pd.isna(rsi_value):
            return np.nan
        elif rsi_value >= 70:
            return '여름'
        elif rsi_value >= 50:
            return '봄'
        elif rsi_value >= 30:
            return '가을'
        else:
            return '겨울'
    
    return rsi.apply(get_season)

def align_data(sp500_df, rsi_series, start_date, end_date):
    """S&P 500 데이터와 RSI 데이터를 월별로 매칭하여 정렬"""
    # 기간 필터링
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    # 월별 매칭을 위해 년-월 키 생성
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    # 공통 년-월 찾기
    sp500_periods = sorted(list(set(sp500_filtered['year_month'])))
    
    if len(sp500_periods) == 0:
        raise ValueError("지정된 기간에 S&P 500 데이터가 없습니다.")
    
    aligned_data = []
    aligned_rsi = []
    aligned_dates = []
    
    for period in sp500_periods:
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            sp500_date = sp500_month.index[0]
            rsi_value = rsi_month['RSI'].iloc[0]
            
            aligned_dates.append(sp500_date)
            aligned_data.append(sp500_month.drop('year_month', axis=1).iloc[0])
            aligned_rsi.append(rsi_value)
    
    if not aligned_dates:
        raise ValueError("데이터 정렬 후 남은 데이터가 없습니다.")

    sp500_aligned = pd.DataFrame(aligned_data, index=aligned_dates)
    rsi_aligned = pd.Series(aligned_rsi, index=aligned_dates)
    
    return sp500_aligned, rsi_aligned

def run_momentum_only_strategy(df, seasons, initial_capital=10000000):
    """모든 계절을 모멘텀으로만 구성한 전략 실행"""
    
    # 모멘텀 온리 전략: 모든 계절에서 Momentum 사용
    momentum_strategy = {
        '여름': 'S&P500 Momentum',
        '봄': 'S&P500 Momentum',
        '가을': 'S&P500 Momentum',
        '겨울': 'S&P500 Momentum'
    }
    
    portfolio_values = []
    transactions = []
    current_style = 'S&P500 Momentum'  # 항상 모멘텀
    current_shares = 0
    cash = 0
    
    for i, date in enumerate(df.index):
        season = seasons.loc[date] if date in seasons.index else np.nan
        
        if i == 0:
            # 초기 투자
            current_price = df.loc[date, current_style]
            current_shares = initial_capital / current_price
            cash = 0
            portfolio_value = initial_capital
            
            transactions.append({
                'date': date,
                'season': season,
                'action': '초기투자',
                'style': current_style,
                'shares': current_shares,
                'price': current_price,
                'value': initial_capital
            })
            
        else:
            # 모멘텀 온리이므로 매매 없이 홀드
            current_price = df.loc[date, current_style]
            portfolio_value = current_shares * current_price
        
        portfolio_values.append(portfolio_value)
    
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    return portfolio_series, transactions

def test_momentum_only():
    """모멘텀 온리 전략 테스트 (CSV와 동일한 기간)"""
    
    print("=== 모멘텀 온리 전략 백테스팅 ===")
    
    # 데이터 로드
    sp500_df = load_sp500_data()
    rsi_series = load_rsi_data()
    
    # 기간 설정 (1999-01 ~ 2001-07)
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2001-07-31')
    
    print(f"백테스팅 기간: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")
    
    # 데이터 정렬
    sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
    
    # 계절 분류
    seasons = classify_market_season(rsi_aligned)
    
    print(f"총 데이터 포인트: {len(sp500_aligned)}개")
    print(f"계절 분포: {seasons.value_counts().to_dict()}")
    
    # 초기 투자금
    initial_capital = 10000000  # 1천만원
    
    # 백테스팅 실행
    portfolio_series, transactions = run_momentum_only_strategy(
        sp500_aligned, seasons, initial_capital
    )
    
    # 결과 계산
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    print(f"\n=== 백테스팅 결과 ===")
    print(f"초기 투자금: {initial_capital:,}원")
    print(f"최종 가치: {final_value:,.0f}원")
    print(f"총 수익률: {total_return:.2%}")
    
    # 상세 결과
    print(f"\n=== 상세 정보 ===")
    print(f"시작일: {portfolio_series.index[0].strftime('%Y-%m-%d')}")
    print(f"종료일: {portfolio_series.index[-1].strftime('%Y-%m-%d')}")
    
    # 기간별 성과
    start_value = portfolio_series.iloc[0]
    end_value = portfolio_series.iloc[-1]
    
    # 모멘텀 지수 값 변화
    momentum_start = sp500_aligned.iloc[0]['S&P500 Momentum']
    momentum_end = sp500_aligned.iloc[-1]['S&P500 Momentum']
    momentum_return = (momentum_end / momentum_start) - 1
    
    print(f"\nS&P500 Momentum 지수:")
    print(f"시작 값: {momentum_start:.3f}")
    print(f"종료 값: {momentum_end:.3f}")
    print(f"지수 수익률: {momentum_return:.2%}")
    
    # 월별 포트폴리오 가치 출력
    print(f"\n=== 월별 포트폴리오 가치 ===")
    for i, (date, value) in enumerate(portfolio_series.items()):
        season = seasons.loc[date] if date in seasons.index else 'N/A'
        rsi_val = rsi_aligned.loc[date] if date in rsi_aligned.index else 'N/A'
        momentum_val = sp500_aligned.loc[date]['S&P500 Momentum']
        
        print(f"{date.strftime('%Y-%m')}: {value:,.0f}원 (RSI:{rsi_val}, 계절:{season}, 모멘텀:{momentum_val:.3f})")
    
    return portfolio_series, total_return

if __name__ == "__main__":
    portfolio_series, total_return = test_momentum_only()