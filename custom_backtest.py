import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import seaborn as sns

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_sp500_data(file_path):
    """S&P 500 스타일 지수 데이터를 로드합니다."""
    try:
        df = pd.read_excel(file_path)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"S&P 500 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        print(f"사용 가능한 지수: {list(df.columns)}")
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

def calculate_comprehensive_metrics(portfolio_series, initial_capital, season_stats):
    """포괄적인 성과 지표를 계산합니다."""
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    start_date = portfolio_series.index[0]
    end_date = portfolio_series.index[-1]
    investment_years = (end_date - start_date).days / 365.25
    
    cagr = (final_value / initial_capital) ** (1 / investment_years) - 1 if investment_years > 0 else 0
    
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max) - 1
    mdd = drawdown.min()
    
    returns = portfolio_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(12)
    
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # 연도별 수익률 계산
    annual_returns = portfolio_series.resample('Y').last().pct_change().dropna()
    annual_performance = {}
    for year, ret in annual_returns.items():
        annual_performance[year.year] = ret
    
    # 최고/최저 연도 찾기
    best_year = annual_returns.idxmax().year if len(annual_returns) > 0 else None
    worst_year = annual_returns.idxmin().year if len(annual_returns) > 0 else None
    best_return = annual_returns.max() if len(annual_returns) > 0 else 0
    worst_return = annual_returns.min() if len(annual_returns) > 0 else 0
    
    # 양수/음수 연도 비율
    positive_years = len(annual_returns[annual_returns > 0])
    total_years = len(annual_returns)
    win_rate_annual = positive_years / total_years if total_years > 0 else 0
    
    season_performance = {}
    for season, returns_list in season_stats.items():
        if returns_list:
            season_performance[season] = {
                'avg_return': np.mean(returns_list),
                'win_rate': len([r for r in returns_list if r > 0]) / len(returns_list),
                'total_periods': len(returns_list)
            }
        else:
            season_performance[season] = {
                'avg_return': 0,
                'win_rate': 0,
                'total_periods': 0
            }
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'investment_years': investment_years,
        'season_performance': season_performance,
        'annual_performance': annual_performance,
        'best_year': best_year,
        'worst_year': worst_year,
        'best_return': best_return,
        'worst_return': worst_return,
        'win_rate_annual': win_rate_annual,
        'total_years': total_years
    }

def print_results(metrics, strategy_name, transactions):
    """백테스팅 결과를 출력합니다."""
    print("\n" + "="*80)
    print(f"           {strategy_name} 백테스팅 결과")
    print("="*80)
    
    print(f"투자 기간: {metrics['investment_years']:.1f}년")
    print(f"총 수익률: {metrics['total_return']:+.2%}")
    print(f"연평균 수익률 (CAGR): {metrics['cagr']:+.2%}")
    print(f"최대 낙폭 (MDD): {metrics['mdd']:.2%}")
    print(f"변동성: {metrics['volatility']:.2%}")
    print(f"샤프 비율: {metrics['sharpe_ratio']:.2f}")
    print(f"연간 승률: {metrics['win_rate_annual']:.1%}")
    
    if metrics.get('best_year') and metrics.get('worst_year'):
        print(f"최고 연도: {metrics['best_year']}년 ({metrics['best_return']:+.1%})")
        print(f"최악 연도: {metrics['worst_year']}년 ({metrics['worst_return']:+.1%})")
    
    # 계절별 성과
    print(f"\n=== 계절별 성과 ===")
    season_perf = metrics['season_performance']
    season_icons = {'봄': '[봄]', '여름': '[여름]', '가을': '[가을]', '겨울': '[겨울]'}
    
    print(f"{'계절':<8} {'평균수익률':<12} {'승률':<8} {'거래횟수':<8}")
    print("-" * 40)
    for season in ['봄', '여름', '가을', '겨울']:
        perf = season_perf[season]
        icon = season_icons[season]
        print(f"{icon} {season:<6} {perf['avg_return']:>10.2%} "
              f"{perf['win_rate']:>8.1%} {perf['total_periods']:>8d}")
    
    # 거래 내역 요약
    print(f"\n=== 거래 내역 요약 ===")
    print(f"총 거래 횟수: {len(transactions)}회")
    
    buy_transactions = [t for t in transactions if t['action'] in ['매수', '초기투자']]
    sell_transactions = [t for t in transactions if t['action'] == '매도']
    
    print(f"매수 거래: {len(buy_transactions)}회")
    print(f"매도 거래: {len(sell_transactions)}회")

def create_chart(portfolio_series, transactions, strategy_name, seasons):
    """백테스팅 결과 차트를 생성합니다."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. 포트폴리오 가치 변화
    ax1 = axes[0, 0]
    ax1.plot(portfolio_series.index, portfolio_series.values, linewidth=2, color='blue')
    ax1.set_title(f'{strategy_name} - 포트폴리오 가치 변화', fontsize=14, fontweight='bold')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 2. 계절별 시장 분포
    ax2 = axes[0, 1]
    season_counts = seasons.value_counts()
    colors_season = {'여름': 'red', '봄': 'green', '가을': 'orange', '겨울': 'blue'}
    season_colors = [colors_season.get(season, 'gray') for season in season_counts.index]
    
    ax2.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', 
           colors=season_colors, startangle=90)
    ax2.set_title('RSI 기반 시장 계절 분포', fontsize=14, fontweight='bold')
    
    # 3. 전략 구성
    ax3 = axes[1, 0]
    strategy_rules = {
        '봄': 'S&P500 Quality',
        '여름': 'S&P500 Momentum', 
        '가을': 'S&P500 Low Volatiltiy Index',
        '겨울': 'S&P500 Low Volatiltiy Index'
    }
    
    seasons_list = ['봄', '여름', '가을', '겨울']
    season_icons = {'봄': '[봄]', '여름': '[여름]', '가을': '[가을]', '겨울': '[겨울]'}
    
    ax3.axis('off')
    ax3.set_title(f'{strategy_name} 구성', fontsize=14, fontweight='bold')
    
    y_pos = 0.8
    for season in seasons_list:
        style = strategy_rules[season]
        short_style = style.replace('S&P500 ', '').replace('S&P 500 ', '').replace(' Index', '')
        ax3.text(0.1, y_pos, f"{season_icons[season]} {season}: {short_style}", 
                fontsize=12, transform=ax3.transAxes)
        y_pos -= 0.15
    
    # 4. 월별 수익률 분포
    ax4 = axes[1, 1]
    monthly_returns = portfolio_series.pct_change().dropna()
    ax4.hist(monthly_returns * 100, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('월별 수익률 분포', fontsize=14, fontweight='bold')
    ax4.set_xlabel('월별 수익률 (%)')
    ax4.set_ylabel('빈도')
    ax4.grid(True, alpha=0.3)
    
    # 5. 누적 수익률
    ax5 = axes[2, 0]
    cumulative_returns = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
    ax5.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, color='green')
    ax5.set_title('누적 수익률', fontsize=14, fontweight='bold')
    ax5.set_ylabel('누적 수익률 (%)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 드로우다운
    ax6 = axes[2, 1]
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max - 1) * 100
    ax6.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax6.plot(drawdown.index, drawdown.values, linewidth=1, color='red')
    ax6.set_title('드로우다운', fontsize=14, fontweight='bold')
    ax6.set_ylabel('드로우다운 (%)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{strategy_name.replace(" ", "_")}_백테스팅_결과.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n백테스팅 차트가 저장되었습니다: {strategy_name.replace(' ', '_')}_백테스팅_결과.png")

def main():
    """메인 함수"""
    print("=== 커스텀 RSI 기반 스타일 로테이션 전략 백테스팅 ===")
    
    # 데이터 로딩
    sp500_df = load_sp500_data('sp500_data.xlsx')
    if sp500_df is None:
        return
    
    rsi_series = load_rsi_data('RSI_DATE.xlsx')
    if rsi_series is None:
        return
    
    # 전략 설정
    strategy_name = "퀄리티-모멘텀-로볼 전략"
    strategy_rules = {
        '봄': 'S&P500 Quality',
        '여름': 'S&P500 Momentum', 
        '가을': 'S&P500 Low Volatiltiy Index',
        '겨울': 'S&P500 Low Volatiltiy Index'
    }
    
    print(f"\n전략 구성:")
    for season, style in strategy_rules.items():
        print(f"  {season}: {style}")
    
    # 백테스팅 파라미터
    start_date = pd.Timestamp(1999, 1, 1)
    end_date = pd.Timestamp(2025, 6, 30)
    initial_capital = 10000000  # 1천만원
    
    print(f"\n백테스팅 기간: {start_date.strftime('%Y년 %m월')} ~ {end_date.strftime('%Y년 %m월')}")
    print(f"초기 투자금: {initial_capital:,}원")
    
    try:
        # 데이터 정렬
        sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
        
        # 계절 분류
        seasons = classify_market_season(rsi_aligned)
        valid_seasons = seasons.dropna()
        
        print(f"계절 분류 완료: {len(valid_seasons)}개 데이터 포인트")
        
        # 백테스팅 실행
        portfolio_series, transactions, season_stats = run_rotation_strategy(
            sp500_aligned, strategy_name, strategy_rules, seasons, initial_capital
        )
        
        # 성과 지표 계산
        metrics = calculate_comprehensive_metrics(portfolio_series, initial_capital, season_stats)
        
        # 결과 출력
        print_results(metrics, strategy_name, transactions)
        
        # 차트 생성
        create_chart(portfolio_series, transactions, strategy_name, valid_seasons)
        
        print(f"\n[SUCCESS] '{strategy_name}' 백테스팅이 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()