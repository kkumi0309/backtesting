"""
S&P 500 RSI 기반 계절별 스타일 로테이션 전략 백테스팅 프로그램

RSI 지표를 기반으로 시장을 4계절로 분류하고,
각 계절에 맞는 스타일별 로테이션 전략의 성과를 비교 분석합니다.

계절 분류:
- 여름 (RSI >= 70): 과매수 구간
- 봄 (50 <= RSI < 70): 상승 구간  
- 가을 (30 <= RSI < 50): 하락 구간
- 겨울 (RSI < 30): 과매도 구간
"""

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

def load_data(file_path):
    """
    Excel 파일에서 S&P 500 지수 데이터를 로드하고 전처리합니다.
    
    Args:
        file_path (str): Excel 파일 경로
        
    Returns:
        pd.DataFrame: 날짜가 인덱스로 설정된 데이터프레임
    """
    try:
        df = pd.read_excel(file_path)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        print(f"데이터 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return None

def calculate_rsi(prices, period=14):
    """
    RSI(Relative Strength Index)를 계산합니다.
    
    Args:
        prices (pd.Series): 가격 시계열
        period (int): RSI 계산 기간 (기본값: 14)
        
    Returns:
        pd.Series: RSI 값
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def classify_market_season(rsi):
    """
    RSI 값을 기반으로 시장 계절을 분류합니다.
    
    Args:
        rsi (pd.Series): RSI 값
        
    Returns:
        pd.Series: 계절 분류 ('여름', '봄', '가을', '겨울')
    """
    def get_season(rsi_value):
        if pd.isna(rsi_value):
            return np.nan
        elif rsi_value >= 70:
            return '여름'  # 과매수
        elif rsi_value >= 50:
            return '봄'    # 상승
        elif rsi_value >= 30:
            return '가을'  # 하락
        else:
            return '겨울'  # 과매도
    
    return rsi.apply(get_season)

def define_rotation_strategies():
    """
    다양한 로테이션 전략을 정의합니다.
    
    Returns:
        dict: 전략명과 계절별 선호 스타일 매핑
    """
    strategies = {
        '모멘텀 전략': {
            '여름': 'S&P500 Momentum',     # 과매수 시 모멘텀
            '봄': 'S&P500 Growth',         # 상승 시 성장
            '가을': 'S&P500 Quality',      # 하락 시 품질
            '겨울': 'S&P500 Value'         # 과매도 시 가치
        },
        '안정성 전략': {
            '여름': 'S&P500 Low Volatiltiy Index',  # 과매수 시 저변동성
            '봄': 'S&P500 Quality',                 # 상승 시 품질
            '가을': 'S&P500 Div Aristocrt TR Index', # 하락 시 배당
            '겨울': 'S&P500 Value'                  # 과매도 시 가치
        },
        '성장 중심 전략': {
            '여름': 'S&P500 Growth',       # 과매수 시 성장
            '봄': 'S&P500 Momentum',       # 상승 시 모멘텀
            '가을': 'S&P500 Growth',       # 하락 시에도 성장
            '겨울': 'S&P500 Quality'       # 과매도 시 품질
        },
        '가치 중심 전략': {
            '여름': 'S&P500 Value',        # 과매수 시 가치
            '봄': 'S&P500 Value',          # 상승 시도 가치
            '가을': 'S&P500 Value',        # 하락 시 가치
            '겨울': 'S&P500 Div Aristocrt TR Index'  # 과매도 시 배당
        },
        '배당 중심 전략': {
            '여름': 'S&P500 Div Aristocrt TR Index',  # 과매수 시 배당
            '봄': 'S&P500 Quality',                   # 상승 시 품질
            '가을': 'S&P500 Div Aristocrt TR Index',  # 하락 시 배당
            '겨울': 'S&P500 Value'                    # 과매도 시 가치
        }
    }
    
    return strategies

def run_rotation_strategy(df, strategy_name, strategy_rules, seasons, initial_capital=10000000):
    """
    로테이션 전략을 실행합니다.
    
    Args:
        df (pd.DataFrame): 스타일 지수 데이터
        strategy_name (str): 전략 명칭
        strategy_rules (dict): 계절별 스타일 규칙
        seasons (pd.Series): 계절 분류
        initial_capital (float): 초기 자본
        
    Returns:
        tuple: (포트폴리오 가치 시계열, 거래 내역, 계절별 통계)
    """
    portfolio_values = []
    transactions = []
    current_style = None
    current_shares = 0
    cash = 0
    
    # 계절별 통계
    season_stats = {'여름': [], '봄': [], '가을': [], '겨울': []}
    
    for i, date in enumerate(df.index):
        season = seasons.loc[date] if date in seasons.index else np.nan
        
        if pd.isna(season):
            # 계절이 정의되지 않은 경우 이전 포트폴리오 유지
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
                # 첫 거래: 전액 투자
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
                # 스타일 변경: 전량 매도 후 새 스타일 매수
                if current_style and current_shares > 0:
                    # 기존 스타일 전량 매도
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
                
                # 새 스타일 전량 매수
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
                # 스타일 유지
                if current_style and current_shares > 0:
                    portfolio_value = current_shares * df.loc[date, current_style]
                else:
                    portfolio_value = cash
            
            # 계절별 수익률 기록
            if i > 0:
                prev_value = portfolio_values[-1]
                period_return = (portfolio_value / prev_value) - 1
                season_stats[season].append(period_return)
        
        portfolio_values.append(portfolio_value)
    
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    
    return portfolio_series, transactions, season_stats

def calculate_comprehensive_metrics(portfolio_series, initial_capital, season_stats):
    """
    포괄적인 성과 지표를 계산합니다.
    """
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    # 투자 기간
    start_date = portfolio_series.index[0]
    end_date = portfolio_series.index[-1]
    investment_years = (end_date - start_date).days / 365.25
    
    # CAGR
    cagr = (final_value / initial_capital) ** (1 / investment_years) - 1 if investment_years > 0 else 0
    
    # MDD
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max) - 1
    mdd = drawdown.min()
    
    # 변동성 (연율화)
    returns = portfolio_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(12)  # 월별 데이터 기준
    
    # 샤프 비율 (무위험 수익률 2% 가정)
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # 계절별 통계
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
        'season_performance': season_performance
    }

def run_all_strategies(df, seasons, initial_capital=10000000):
    """
    모든 로테이션 전략을 실행하고 결과를 비교합니다.
    """
    strategies = define_rotation_strategies()
    results = {}
    
    print("\n=== 로테이션 전략 백테스팅 실행 ===")
    
    for strategy_name, strategy_rules in strategies.items():
        print(f"\n{strategy_name} 실행 중...")
        
        portfolio_series, transactions, season_stats = run_rotation_strategy(
            df, strategy_name, strategy_rules, seasons, initial_capital
        )
        
        metrics = calculate_comprehensive_metrics(portfolio_series, initial_capital, season_stats)
        
        results[strategy_name] = {
            'portfolio_series': portfolio_series,
            'transactions': transactions,
            'metrics': metrics,
            'strategy_rules': strategy_rules
        }
        
        print(f"  최종 수익률: {metrics['total_return']:.2%}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  샤프 비율: {metrics['sharpe_ratio']:.2f}")
    
    return results

def create_comparison_chart(results, market_data, seasons):
    """
    전략별 성과 비교 차트를 생성합니다.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. 포트폴리오 가치 변화
    ax1 = axes[0, 0]
    for strategy_name, result in results.items():
        ax1.plot(result['portfolio_series'].index, result['portfolio_series'].values, 
                label=strategy_name, linewidth=2)
    
    ax1.set_title('전략별 포트폴리오 가치 변화', fontsize=14, fontweight='bold')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 2. 누적 수익률 비교
    ax2 = axes[0, 1]
    for strategy_name, result in results.items():
        portfolio_series = result['portfolio_series']
        cumulative_returns = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
        ax2.plot(cumulative_returns.index, cumulative_returns.values, 
                label=strategy_name, linewidth=2)
    
    ax2.set_title('전략별 누적 수익률 비교', fontsize=14, fontweight='bold')
    ax2.set_ylabel('누적 수익률 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 주요 성과 지표 비교 (막대 차트)
    ax3 = axes[1, 0]
    strategies = list(results.keys())
    cagr_values = [results[s]['metrics']['cagr'] * 100 for s in strategies]
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    bars = ax3.bar(strategies, cagr_values, color=colors)
    ax3.set_title('전략별 연평균 수익률 (CAGR)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('CAGR (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 막대 위에 값 표시
    for bar, value in zip(bars, cagr_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 4. 리스크 지표 비교
    ax4 = axes[1, 1]
    volatility_values = [results[s]['metrics']['volatility'] * 100 for s in strategies]
    mdd_values = [abs(results[s]['metrics']['mdd']) * 100 for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax4.bar(x - width/2, volatility_values, width, label='변동성', alpha=0.8)
    ax4.bar(x + width/2, mdd_values, width, label='최대낙폭', alpha=0.8)
    
    ax4.set_title('전략별 리스크 지표', fontsize=14, fontweight='bold')
    ax4.set_ylabel('비율 (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45)
    ax4.legend()
    
    # 5. 계절별 시장 분포
    ax5 = axes[2, 0]
    season_counts = seasons.value_counts()
    colors_season = {'여름': 'red', '봄': 'green', '가을': 'orange', '겨울': 'blue'}
    season_colors = [colors_season.get(season, 'gray') for season in season_counts.index]
    
    ax5.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', 
           colors=season_colors, startangle=90)
    ax5.set_title('RSI 기반 시장 계절 분포', fontsize=14, fontweight='bold')
    
    # 6. 샤프 비율 비교
    ax6 = axes[2, 1]
    sharpe_values = [results[s]['metrics']['sharpe_ratio'] for s in strategies]
    bars = ax6.bar(strategies, sharpe_values, color=colors)
    ax6.set_title('전략별 샤프 비율', fontsize=14, fontweight='bold')
    ax6.set_ylabel('샤프 비율')
    ax6.tick_params(axis='x', rotation=45)
    
    # 막대 위에 값 표시
    for bar, value in zip(bars, sharpe_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rotation_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n전략 비교 차트가 'rotation_strategy_comparison.png' 파일로 저장되었습니다.")

def print_detailed_results(results):
    """
    상세한 결과를 출력합니다.
    """
    print("\n" + "="*80)
    print("                    RSI 기반 로테이션 전략 성과 비교")
    print("="*80)
    
    # 성과 요약 테이블
    print(f"{'전략명':<15} {'총수익률':<10} {'CAGR':<8} {'변동성':<8} {'MDD':<8} {'샤프비율':<8}")
    print("-" * 80)
    
    for strategy_name, result in results.items():
        metrics = result['metrics']
        print(f"{strategy_name:<15} "
              f"{metrics['total_return']:>8.1%} "
              f"{metrics['cagr']:>8.1%} "
              f"{metrics['volatility']:>8.1%} "
              f"{metrics['mdd']:>8.1%} "
              f"{metrics['sharpe_ratio']:>8.2f}")
    
    print("="*80)
    
    # 최고 성과 전략
    best_return = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    
    print(f"\n[BEST] 최고 수익률: {best_return[0]} ({best_return[1]['metrics']['total_return']:.2%})")
    print(f"[TOP] 최고 샤프비율: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.2f})")
    
    # 계절별 성과 (최고 수익률 전략 기준)
    print(f"\n=== {best_return[0]} 계절별 성과 ===")
    season_perf = best_return[1]['metrics']['season_performance']
    print(f"{'계절':<6} {'평균수익률':<12} {'승률':<8} {'거래횟수':<8}")
    print("-" * 40)
    for season in ['여름', '봄', '가을', '겨울']:
        perf = season_perf[season]
        print(f"{season:<6} {perf['avg_return']:>10.2%} "
              f"{perf['win_rate']:>8.1%} {perf['total_periods']:>8d}")

def get_user_input():
    """
    사용자로부터 백테스팅 파라미터를 입력받습니다.
    """
    print("\n=== RSI 기반 스타일 로테이션 전략 백테스팅 ===")
    
    try:
        start_year = int(input("\n백테스팅 시작 연도 (YYYY): "))
        start_month = int(input("백테스팅 시작 월 (1-12): "))
        end_year = int(input("백테스팅 종료 연도 (YYYY): "))
        end_month = int(input("백테스팅 종료 월 (1-12): "))
        initial_capital = float(input("초기 투자 원금 (원): "))
        rsi_period = int(input("RSI 계산 기간 (기본값 14): ") or "14")
        
        return {
            'start_year': start_year,
            'start_month': start_month,
            'end_year': end_year,
            'end_month': end_month,
            'initial_capital': initial_capital,
            'rsi_period': rsi_period
        }
    
    except (ValueError, KeyboardInterrupt) as e:
        print(f"입력 오류: {e}")
        return None

def main():
    """
    메인 함수 - RSI 기반 로테이션 전략 백테스팅을 실행합니다.
    """
    # 1. 데이터 로딩
    df = load_data('sp500_data.xlsx')
    if df is None:
        return
    
    # 2. 사용자 입력
    user_input = get_user_input()
    if user_input is None:
        return
    
    try:
        # 3. 데이터 필터링
        start_date = pd.Timestamp(user_input['start_year'], user_input['start_month'], 1)
        end_date = pd.Timestamp(user_input['end_year'], user_input['end_month'], 28)
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df.loc[mask].copy()
        
        if len(filtered_df) == 0:
            raise ValueError("지정된 기간에 해당하는 데이터가 없습니다.")
        
        print(f"\n필터링된 데이터: {len(filtered_df)}개 데이터 포인트")
        
        # 4. RSI 계산 (S&P500 Growth 기준)
        print(f"\nRSI 계산 중... (기간: {user_input['rsi_period']})")
        sp500_prices = filtered_df['S&P500 Growth']
        rsi = calculate_rsi(sp500_prices, user_input['rsi_period'])
        
        # 5. 계절 분류
        seasons = classify_market_season(rsi)
        valid_seasons = seasons.dropna()
        
        print(f"계절 분류 완료: {len(valid_seasons)}개 데이터 포인트")
        print("계절별 분포:")
        for season, count in valid_seasons.value_counts().items():
            print(f"  {season}: {count}회 ({count/len(valid_seasons)*100:.1f}%)")
        
        # 6. 모든 로테이션 전략 실행
        results = run_all_strategies(filtered_df, seasons, user_input['initial_capital'])
        
        # 7. 결과 출력
        print_detailed_results(results)
        
        # 8. 시각화
        create_comparison_chart(results, filtered_df, valid_seasons)
        
        print("\n[SUCCESS] RSI 기반 로테이션 전략 분석이 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()