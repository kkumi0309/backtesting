import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# AA.py에서 import한 함수들을 복사
def load_sp500_data(file_path):
    """S&P 500 스타일 지수 데이터를 로드합니다."""
    try:
        df = pd.read_excel(file_path)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"S&P 500 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        return df
    except Exception as e:
        print(f"S&P 500 데이터 로딩 오류: {e}")
        return None

def load_rsi_data(file_path):
    """기존에 계산된 RSI 데이터를 로드합니다."""
    try:
        df = pd.read_excel(file_path, skiprows=1)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        rsi_series = df['RSI'].dropna()
        print(f"RSI 데이터 로딩 완료: {len(rsi_series)}개 데이터 포인트")
        return rsi_series
    except Exception as e:
        print(f"RSI 데이터 로딩 오류: {e}")
        return None

def classify_market_season(rsi):
    """RSI 값을 기반으로 시장 계절을 분류합니다."""
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
    """S&P 500 데이터와 RSI 데이터를 월별로 매칭하여 정렬합니다."""
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
    
    print(f"데이터 정렬 완료: {len(aligned_dates)}개 데이터 포인트")
    return sp500_aligned, rsi_aligned

def run_rotation_strategy(df, strategy_name, strategy_rules, seasons, initial_capital=10000000):
    """로테이션 전략을 실행합니다."""
    portfolio_values = []
    transactions = []
    current_style = None
    current_shares = 0
    cash = 0
    
    season_stats = {'여름': [], '봄': [], '가을': [], '겨울': []}
    
    for i, date in enumerate(df.index):
        season = seasons.loc[date] if date in seasons.index else np.nan
        
        if pd.isna(season):
            if i == 0:
                portfolio_value = initial_capital
                cash = initial_capital
            else:
                if current_style and current_shares > 0:
                    portfolio_value = cash + (current_shares * df.loc[date, current_style])
                else:
                    portfolio_value = cash
        else:
            target_style = strategy_rules[season]
            
            if i == 0:
                current_style = target_style
                current_price = df.loc[date, current_style]
                current_shares = initial_capital / current_price
                cash = 0
                portfolio_value = initial_capital
                
                transactions.append({
                    'date': date,
                    'season': season,
                    'action': '초기투자',
                    'from_style': None,
                    'to_style': current_style,
                    'shares': current_shares,
                    'price': current_price,
                    'value': initial_capital
                })
            
            elif target_style != current_style:
                if current_style and current_shares > 0:
                    sell_price = df.loc[date, current_style]
                    cash = current_shares * sell_price
                    
                    transactions.append({
                        'date': date,
                        'season': season,
                        'action': '매도',
                        'from_style': current_style,
                        'to_style': None,
                        'shares': current_shares,
                        'price': sell_price,
                        'value': cash
                    })
                
                current_style = target_style
                buy_price = df.loc[date, current_style]
                current_shares = cash / buy_price
                cash = 0
                
                transactions.append({
                    'date': date,
                    'season': season,
                    'action': '매수',
                    'from_style': None,
                    'to_style': current_style,
                    'shares': current_shares,
                    'price': buy_price,
                    'value': current_shares * buy_price
                })
                
                portfolio_value = current_shares * buy_price
            
            else:
                if current_style and current_shares > 0:
                    portfolio_value = current_shares * df.loc[date, current_style]
                else:
                    portfolio_value = cash
            
            if i > 0:
                prev_value = portfolio_values[-1]
                period_return = (portfolio_value / prev_value) - 1
                season_stats[season].append(period_return)
        
        portfolio_values.append(portfolio_value)
    
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    return portfolio_series, transactions, season_stats

def test_aa_momentum_strategy():
    """AA.py 방식으로 모멘텀 전략 테스트"""
    
    print("=== AA.py 방식 모멘텀 전략 테스트 ===")
    
    # 1. 데이터 로딩
    sp500_df = load_sp500_data('sp500_data.xlsx')
    if sp500_df is None:
        return
    
    rsi_series = load_rsi_data('RSI_DATE.xlsx')
    if rsi_series is None:
        return
    
    # 2. 기간 설정 (1999-2025)
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2025-06-30')
    
    print(f"백테스팅 기간: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")
    
    # 3. 데이터 정렬
    sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
    
    # 4. 계절 분류
    seasons = classify_market_season(rsi_aligned)
    
    print(f"계절 분포: {seasons.value_counts().to_dict()}")
    
    # 5. 모멘텀 온리 전략 설정
    momentum_strategy = {
        '여름': 'S&P500 Momentum',
        '봄': 'S&P500 Momentum',
        '가을': 'S&P500 Momentum',
        '겨울': 'S&P500 Momentum'
    }
    
    # 6. 백테스팅 실행
    initial_capital = 10000000
    portfolio_series, transactions, season_stats = run_rotation_strategy(
        sp500_aligned, "모멘텀 전략", momentum_strategy, seasons, initial_capital
    )
    
    # 7. 결과 출력
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    start_date_actual = portfolio_series.index[0]
    end_date_actual = portfolio_series.index[-1]
    investment_years = (end_date_actual - start_date_actual).days / 365.25
    cagr = (final_value / initial_capital) ** (1 / investment_years) - 1
    
    print(f"\n=== 백테스팅 결과 ===")
    print(f"초기 투자금: {initial_capital:,}원")
    print(f"최종 가치: {final_value:,.0f}원")
    print(f"총 수익률: {total_return:.1%}")
    print(f"연평균 수익률 (CAGR): {cagr:.2%}")
    print(f"투자 기간: {investment_years:.1f}년")
    
    # 8. 거래 내역 확인
    print(f"\n=== 거래 내역 (총 {len(transactions)}건) ===")
    for i, trans in enumerate(transactions[:5]):
        print(f"{i+1}. {trans['date'].strftime('%Y-%m-%d')}: {trans['action']} - {trans.get('to_style', trans.get('from_style', 'N/A'))}")
    
    if len(transactions) > 5:
        print(f"... (나머지 {len(transactions)-5}건 생략)")
    
    # 9. 모멘텀 지수 성과
    momentum_start = sp500_aligned.iloc[0]['S&P500 Momentum']
    momentum_end = sp500_aligned.iloc[-1]['S&P500 Momentum']
    momentum_return = (momentum_end / momentum_start) - 1
    momentum_cagr = (momentum_end / momentum_start) ** (1 / investment_years) - 1
    
    print(f"\n=== S&P500 Momentum 지수 성과 ===")
    print(f"시작 값: {momentum_start:.3f}")
    print(f"종료 값: {momentum_end:.3f}")
    print(f"지수 총 수익률: {momentum_return:.1%}")
    print(f"지수 CAGR: {momentum_cagr:.2%}")
    
    return portfolio_series

if __name__ == "__main__":
    portfolio = test_aa_momentum_strategy()