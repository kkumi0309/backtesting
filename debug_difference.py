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

def align_data_aa_style(sp500_df, rsi_series, start_date, end_date):
    """AA.py 스타일의 데이터 정렬"""
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
    
    sp500_aligned = pd.DataFrame(aligned_data, index=aligned_dates)
    rsi_aligned = pd.Series(aligned_rsi, index=aligned_dates)
    
    return sp500_aligned, rsi_aligned

def run_rotation_strategy_aa_style(df, strategy_name, strategy_rules, seasons, initial_capital=10000000):
    """AA.py 스타일의 로테이션 전략 실행"""
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

def run_momentum_simple(df, initial_capital=10000000):
    """단순 모멘텀 홀드 전략"""
    momentum_style = 'S&P500 Momentum'
    
    # 초기 주식 수 계산
    initial_price = df.iloc[0][momentum_style]
    shares = initial_capital / initial_price
    
    # 포트폴리오 가치 계산
    portfolio_values = df[momentum_style] * shares
    
    return portfolio_values

def debug_strategies():
    """두 방식의 결과 차이 디버깅"""
    
    print("=== 모멘텀 전략 차이 분석 ===")
    
    # 데이터 로드
    sp500_df = load_sp500_data()
    rsi_series = load_rsi_data()
    
    # 기간 설정
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2025-06-30')
    
    # 데이터 정렬
    sp500_aligned, rsi_aligned = align_data_aa_style(sp500_df, rsi_series, start_date, end_date)
    seasons = classify_market_season(rsi_aligned)
    
    print(f"데이터 포인트: {len(sp500_aligned)}개")
    print(f"기간: {sp500_aligned.index[0]} ~ {sp500_aligned.index[-1]}")
    
    initial_capital = 10000000
    
    # 방법 1: AA.py 스타일 (로테이션 전략으로 모멘텀만)
    momentum_strategy = {
        '여름': 'S&P500 Momentum',
        '봄': 'S&P500 Momentum', 
        '가을': 'S&P500 Momentum',
        '겨울': 'S&P500 Momentum'
    }
    
    portfolio1, transactions1, season_stats1 = run_rotation_strategy_aa_style(
        sp500_aligned, "모멘텀 전략", momentum_strategy, seasons, initial_capital
    )
    
    # 방법 2: 단순 홀드
    portfolio2 = run_momentum_simple(sp500_aligned, initial_capital)
    
    print(f"\n=== 결과 비교 ===")
    print(f"방법 1 (AA.py 로테이션): {portfolio1.iloc[-1]:,.0f}원")
    print(f"방법 2 (단순 홀드): {portfolio2.iloc[-1]:,.0f}원")
    print(f"차이: {abs(portfolio1.iloc[-1] - portfolio2.iloc[-1]):,.0f}원")
    
    # 첫 10개 데이터 포인트 비교
    print(f"\n=== 첫 10개 데이터 포인트 비교 ===")
    print("날짜           방법1        방법2        차이         거래")
    print("-" * 70)
    
    transaction_dates = set([t['date'] for t in transactions1])
    
    for i in range(min(10, len(portfolio1))):
        date = portfolio1.index[i]
        val1 = portfolio1.iloc[i]
        val2 = portfolio2.iloc[i]
        diff = abs(val1 - val2)
        traded = "거래" if date in transaction_dates else ""
        
        print(f"{date.strftime('%Y-%m-%d')} {val1:>12,.0f} {val2:>12,.0f} {diff:>12,.0f} {traded}")
    
    # 거래 내역 확인
    print(f"\n=== 거래 내역 (처음 5개) ===")
    for i, trans in enumerate(transactions1[:5]):
        print(f"{i+1}. {trans['date'].strftime('%Y-%m-%d')}: {trans['action']} - {trans.get('to_style', trans.get('from_style', 'N/A'))}")
    
    # RSI와 계절 변화 확인
    print(f"\n=== 계절 변화 확인 (처음 10개) ===")
    for i in range(min(10, len(seasons))):
        date = seasons.index[i]
        season = seasons.iloc[i]
        rsi_val = rsi_aligned.iloc[i]
        
        print(f"{date.strftime('%Y-%m-%d')}: RSI={rsi_val:.1f}, 계절={season}")
    
    return portfolio1, portfolio2

if __name__ == "__main__":
    portfolio1, portfolio2 = debug_strategies()