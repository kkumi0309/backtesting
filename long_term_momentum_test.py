import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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

def run_momentum_strategy(df, seasons, initial_capital=10000000):
    """모멘텀 온리 전략 실행"""
    
    portfolio_values = []
    current_style = 'S&P500 Momentum'
    current_shares = 0
    
    for i, date in enumerate(df.index):
        if i == 0:
            # 초기 투자
            current_price = df.loc[date, current_style]
            current_shares = initial_capital / current_price
            portfolio_value = initial_capital
        else:
            # 모멘텀 홀드
            current_price = df.loc[date, current_style]
            portfolio_value = current_shares * current_price
        
        portfolio_values.append(portfolio_value)
    
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    return portfolio_series

def calculate_performance_metrics(portfolio_series, initial_capital):
    """성과 지표 계산"""
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    start_date = portfolio_series.index[0]
    end_date = portfolio_series.index[-1]
    investment_years = (end_date - start_date).days / 365.25
    
    # CAGR (연평균 수익률)
    cagr = (final_value / initial_capital) ** (1 / investment_years) - 1 if investment_years > 0 else 0
    
    # MDD (최대 낙폭)
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max) - 1
    mdd = drawdown.min()
    
    # 변동성 (월간 수익률 기준)
    returns = portfolio_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(12)  # 연환산
    
    # 샤프 비율 (무위험 수익률 2% 가정)
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'investment_years': investment_years
    }

def create_visualization(portfolio_series, momentum_series, seasons):
    """성과 시각화"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. 포트폴리오 가치 변화
    ax1 = axes[0]
    ax1.plot(portfolio_series.index, portfolio_series.values, label='모멘텀 전략', linewidth=2, color='blue')
    ax1.set_title('모멘텀 전략 포트폴리오 가치 변화 (1999-2025)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 2. 모멘텀 지수 변화
    ax2 = axes[1]
    ax2.plot(momentum_series.index, momentum_series.values, label='S&P500 Momentum 지수', linewidth=2, color='green')
    ax2.set_title('S&P500 Momentum 지수 변화', fontsize=14, fontweight='bold')
    ax2.set_ylabel('지수 값')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 계절별 분포 (최근 5년)
    ax3 = axes[2]
    recent_seasons = seasons[seasons.index >= '2020-01-01']
    season_counts = recent_seasons.value_counts()
    colors = {'여름': 'red', '봄': 'green', '가을': 'orange', '겨울': 'blue'}
    season_colors = [colors.get(season, 'gray') for season in season_counts.index]
    
    ax3.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', 
           colors=season_colors, startangle=90)
    ax3.set_title('RSI 기반 시장 계절 분포 (2020-2025)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('long_term_momentum_strategy.png', dpi=300, bbox_inches='tight')
    plt.show()

def long_term_momentum_test():
    """장기간 모멘텀 전략 백테스팅 (1999-2025)"""
    
    print("=== 장기간 모멘텀 온리 전략 백테스팅 (1999-2025) ===")
    
    # 데이터 로드
    sp500_df = load_sp500_data()
    rsi_series = load_rsi_data()
    
    # 기간 설정
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2025-06-30')
    
    print(f"백테스팅 기간: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")
    
    # 데이터 정렬
    sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
    
    # 계절 분류
    seasons = classify_market_season(rsi_aligned)
    
    print(f"총 데이터 포인트: {len(sp500_aligned)}개")
    print(f"투자 기간: {(end_date - start_date).days / 365.25:.1f}년")
    
    # 계절별 분포
    season_counts = seasons.value_counts()
    print(f"계절 분포: {season_counts.to_dict()}")
    
    # 초기 투자금
    initial_capital = 10000000  # 1천만원
    
    # 백테스팅 실행
    portfolio_series = run_momentum_strategy(sp500_aligned, seasons, initial_capital)
    
    # 성과 지표 계산
    metrics = calculate_performance_metrics(portfolio_series, initial_capital)
    
    print(f"\n=== 장기간 백테스팅 결과 ===")
    print(f"초기 투자금: {initial_capital:,}원")
    print(f"최종 가치: {metrics['final_value']:,.0f}원")
    print(f"총 수익률: {metrics['total_return']:.1%}")
    print(f"연평균 수익률 (CAGR): {metrics['cagr']:.2%}")
    print(f"최대 낙폭 (MDD): {metrics['mdd']:.2%}")
    print(f"변동성 (연환산): {metrics['volatility']:.2%}")
    print(f"샤프 비율: {metrics['sharpe_ratio']:.2f}")
    print(f"투자 기간: {metrics['investment_years']:.1f}년")
    
    # 모멘텀 지수 성과
    momentum_start = sp500_aligned.iloc[0]['S&P500 Momentum']
    momentum_end = sp500_aligned.iloc[-1]['S&P500 Momentum']
    momentum_total_return = (momentum_end / momentum_start) - 1
    momentum_cagr = (momentum_end / momentum_start) ** (1 / metrics['investment_years']) - 1
    
    print(f"\n=== S&P500 Momentum 지수 성과 ===")
    print(f"시작 값: {momentum_start:.3f}")
    print(f"종료 값: {momentum_end:.3f}")
    print(f"총 수익률: {momentum_total_return:.1%}")
    print(f"연평균 수익률: {momentum_cagr:.2%}")
    
    # 연도별 수익률 분석
    print(f"\n=== 연도별 수익률 ===")
    annual_returns = portfolio_series.resample('Y').last().pct_change().dropna()
    for year, ret in annual_returns.items():
        print(f"{year.year}: {ret:.1%}")
    
    # 시각화
    momentum_series = sp500_aligned['S&P500 Momentum']
    create_visualization(portfolio_series, momentum_series, seasons)
    
    print(f"\n차트가 'long_term_momentum_strategy.png'로 저장되었습니다.")
    
    return portfolio_series, metrics

if __name__ == "__main__":
    portfolio_series, metrics = long_term_momentum_test()