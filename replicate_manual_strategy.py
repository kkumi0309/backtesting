"""
수기 작업 전략을 정확히 재현하는 프로그램

분석 결과를 바탕으로 수기 작업에서 사용한 실제 전략을 
동일하게 재현하여 결과를 비교합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """필요한 데이터를 모두 로드합니다."""
    # S&P 500 스타일 지수 데이터
    sp500_df = pd.read_excel('sp500_data.xlsx')
    date_column = sp500_df.columns[0]
    sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
    sp500_df.set_index(date_column, inplace=True)
    sp500_df.sort_index(inplace=True)
    
    # RSI 데이터
    rsi_df = pd.read_excel('RSI_DATE.xlsx', skiprows=1)
    date_column = rsi_df.columns[0]
    rsi_df[date_column] = pd.to_datetime(rsi_df[date_column])
    rsi_df.set_index(date_column, inplace=True)
    rsi_series = rsi_df['RSI'].dropna()
    
    # 수기 작업 데이터
    manual_df = pd.read_excel('11.xlsx', header=1)
    date_column = manual_df.columns[0]
    manual_df[date_column] = pd.to_datetime(manual_df[date_column])
    manual_df.set_index(date_column, inplace=True)
    manual_df.sort_index(inplace=True)
    
    print("모든 데이터 로딩 완료")
    return sp500_df, rsi_series, manual_df

def extract_manual_strategy_sequence(manual_df):
    """수기 작업에서 실제 사용한 전략 순서를 추출합니다."""
    
    # 숫자 코딩을 스타일명으로 변환
    style_mapping = {
        1: 'S&P500 Growth',
        2: 'S&P500 Value',
        3: 'S&P500 Momentum', 
        4: 'S&P500 Quality',
        5: 'S&P500 Low Volatiltiy Index',
        6: 'S&P500 Div Aristocrt TR Index'
    }
    
    # S&P500 컬럼에서 전략 순서 추출
    if 'S&P500' in manual_df.columns:
        strategy_sequence = manual_df['S&P500'].dropna()
        
        # 숫자를 스타일명으로 변환
        strategy_names = strategy_sequence.map(style_mapping)
        
        print(f"수기 작업 전략 순서 추출 완료: {len(strategy_names)}개 포인트")
        print(f"사용된 전략: {strategy_names.value_counts()}")
        
        return strategy_names
    
    return None

def replicate_manual_backtesting(sp500_df, strategy_sequence, initial_capital=10000000):
    """수기 작업과 동일한 방식으로 백테스팅을 수행합니다."""
    
    print(f"\n=== 수기 작업 재현 백테스팅 시작 ===")
    print(f"초기 자본: {initial_capital:,}원")
    
    portfolio_values = []
    transactions = []
    current_style = None
    current_shares = 0
    cash = 0
    
    # 공통 날짜 찾기
    common_dates = sp500_df.index.intersection(strategy_sequence.index)
    print(f"공통 데이터 포인트: {len(common_dates)}개")
    
    if len(common_dates) == 0:
        print("공통 날짜가 없습니다.")
        return None, None
    
    for i, date in enumerate(common_dates):
        target_style = strategy_sequence[date]
        
        if pd.isna(target_style) or target_style not in sp500_df.columns:
            # 스타일이 정의되지 않거나 없는 경우 이전 포트폴리오 유지
            if i == 0:
                portfolio_value = initial_capital
                cash = initial_capital
            else:
                if current_style and current_shares > 0:
                    portfolio_value = cash + (current_shares * sp500_df.loc[date, current_style])
                else:
                    portfolio_value = cash
        else:
            if i == 0:
                # 첫 거래: 전액 투자
                current_style = target_style
                current_price = sp500_df.loc[date, current_style]
                current_shares = initial_capital / current_price
                cash = 0
                portfolio_value = initial_capital
                
                transactions.append({
                    'date': date,
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
                    sell_price = sp500_df.loc[date, current_style]
                    cash = current_shares * sell_price
                    
                    transactions.append({
                        'date': date,
                        'action': '매도',
                        'from_style': current_style,
                        'to_style': None,
                        'shares': current_shares,
                        'price': sell_price,
                        'value': cash
                    })
                
                # 새 스타일 전량 매수
                current_style = target_style
                buy_price = sp500_df.loc[date, current_style]
                current_shares = cash / buy_price
                cash = 0
                
                transactions.append({
                    'date': date,
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
                    portfolio_value = current_shares * sp500_df.loc[date, current_style]
                else:
                    portfolio_value = cash
        
        portfolio_values.append(portfolio_value)
    
    # 결과를 시리즈로 변환
    portfolio_series = pd.Series(portfolio_values, index=common_dates)
    
    print(f"백테스팅 완료:")
    print(f"  최종 포트폴리오 가치: {portfolio_series.iloc[-1]:,.0f}원")
    print(f"  총 수익률: {(portfolio_series.iloc[-1] / initial_capital - 1)*100:.2f}%")
    print(f"  총 거래 횟수: {len([t for t in transactions if t['action'] != '초기투자'])}회")
    
    return portfolio_series, transactions

def calculate_performance_metrics(portfolio_series, initial_capital):
    """성과 지표를 계산합니다."""
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
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'investment_years': investment_years
    }

def compare_with_program_strategies(replicated_results, sp500_df, rsi_series, initial_capital):
    """프로그램의 기본 전략들과 비교합니다."""
    
    # 월별 데이터 정렬
    from sp500_custom_rotation_strategy import align_data, classify_market_season, run_rotation_strategy
    
    start_date = replicated_results.index.min()
    end_date = replicated_results.index.max()
    
    sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
    seasons = classify_market_season(rsi_aligned)
    
    # 기본 전략들
    basic_strategies = {
        '모멘텀 전략': {
            '여름': 'S&P500 Momentum',
            '봄': 'S&P500 Growth',
            '가을': 'S&P500 Quality',
            '겨울': 'S&P500 Value'
        },
        '배당 중심 전략': {
            '여름': 'S&P500 Div Aristocrt TR Index',
            '봄': 'S&P500 Quality',
            '가을': 'S&P500 Div Aristocrt TR Index',
            '겨울': 'S&P500 Value'
        }
    }
    
    print(f"\n=== 기본 전략들과 비교 ===")
    
    comparison_results = {}
    
    for strategy_name, strategy_rules in basic_strategies.items():
        portfolio_series, transactions, season_stats = run_rotation_strategy(
            sp500_aligned, strategy_name, strategy_rules, seasons, initial_capital
        )
        
        metrics = calculate_performance_metrics(portfolio_series, initial_capital)
        comparison_results[strategy_name] = {
            'portfolio_series': portfolio_series,
            'metrics': metrics
        }
        
        print(f"{strategy_name}:")
        print(f"  총 수익률: {metrics['total_return']:.2%}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
    
    return comparison_results

def create_replication_comparison_chart(replicated_series, comparison_results, initial_capital):
    """재현 결과와 기본 전략들의 비교 차트를 생성합니다."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 포트폴리오 가치 변화 비교
    ax1 = axes[0, 0]
    ax1.plot(replicated_series.index, replicated_series.values, 
            label='수기 작업 재현', linewidth=3, color='red')
    
    for strategy_name, result in comparison_results.items():
        ax1.plot(result['portfolio_series'].index, result['portfolio_series'].values,
                label=strategy_name, linewidth=2, alpha=0.7)
    
    ax1.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5, 
               label=f'초기 자본: {initial_capital:,.0f}원')
    
    ax1.set_title('포트폴리오 가치 변화 비교', fontsize=14, fontweight='bold')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 2. 수익률 비교
    ax2 = axes[0, 1]
    
    replicated_return = (replicated_series.iloc[-1] / initial_capital - 1) * 100
    strategy_returns = [(name, (result['metrics']['total_return'] * 100)) 
                       for name, result in comparison_results.items()]
    
    all_strategies = [('수기 작업 재현', replicated_return)] + strategy_returns
    names = [item[0] for item in all_strategies]
    returns = [item[1] for item in all_strategies]
    
    colors = ['red'] + ['blue', 'green', 'orange'][:len(strategy_returns)]
    bars = ax2.bar(names, returns, color=colors, alpha=0.7)
    
    ax2.set_title('총 수익률 비교', fontsize=14, fontweight='bold')
    ax2.set_ylabel('총 수익률 (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 막대 위에 값 표시
    for bar, value in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 3. 누적 수익률 변화
    ax3 = axes[1, 0]
    
    replicated_cumret = (replicated_series / initial_capital - 1) * 100
    ax3.plot(replicated_cumret.index, replicated_cumret.values,
            label='수기 작업 재현', linewidth=3, color='red')
    
    for strategy_name, result in comparison_results.items():
        portfolio_series = result['portfolio_series']
        cumret = (portfolio_series / initial_capital - 1) * 100
        ax3.plot(cumret.index, cumret.values, label=strategy_name, linewidth=2, alpha=0.7)
    
    ax3.set_title('누적 수익률 변화', fontsize=14, fontweight='bold')
    ax3.set_ylabel('누적 수익률 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 성과 지표 요약
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 재현 결과 성과 지표
    replicated_metrics = calculate_performance_metrics(replicated_series, initial_capital)
    
    summary_text = "=== 성과 요약 ===\n\n"
    summary_text += "수기 작업 재현:\n"
    summary_text += f"  총 수익률: {replicated_metrics['total_return']:.2%}\n"
    summary_text += f"  CAGR: {replicated_metrics['cagr']:.2%}\n"
    summary_text += f"  MDD: {replicated_metrics['mdd']:.2%}\n"
    summary_text += f"  샤프 비율: {replicated_metrics['sharpe_ratio']:.2f}\n\n"
    
    for strategy_name, result in comparison_results.items():
        metrics = result['metrics']
        summary_text += f"{strategy_name}:\n"
        summary_text += f"  총 수익률: {metrics['total_return']:.2%}\n"
        summary_text += f"  CAGR: {metrics['cagr']:.2%}\n"
        summary_text += f"  MDD: {metrics['mdd']:.2%}\n"
        summary_text += f"  샤프 비율: {metrics['sharpe_ratio']:.2f}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('manual_replication_comparison.png', dpi=300, bbox_inches='tight')
    print("\n재현 비교 차트가 'manual_replication_comparison.png'로 저장되었습니다.")

def main():
    """메인 함수"""
    print("=== 수기 작업 전략 정확 재현 프로그램 ===")
    
    # 1. 데이터 로딩
    sp500_df, rsi_series, manual_df = load_data()
    
    # 2. 수기 작업 전략 순서 추출
    strategy_sequence = extract_manual_strategy_sequence(manual_df)
    
    if strategy_sequence is None:
        print("수기 작업 전략을 추출할 수 없습니다.")
        return
    
    # 3. 수기 작업 재현 백테스팅
    initial_capital = 10000000
    replicated_series, transactions = replicate_manual_backtesting(
        sp500_df, strategy_sequence, initial_capital
    )
    
    if replicated_series is None:
        print("백테스팅 재현에 실패했습니다.")
        return
    
    # 4. 성과 지표 계산
    replicated_metrics = calculate_performance_metrics(replicated_series, initial_capital)
    
    print(f"\n=== 수기 작업 재현 결과 ===")
    print(f"총 수익률: {replicated_metrics['total_return']:.2%}")
    print(f"CAGR: {replicated_metrics['cagr']:.2%}")
    print(f"MDD: {replicated_metrics['mdd']:.2%}")
    print(f"샤프 비율: {replicated_metrics['sharpe_ratio']:.2f}")
    print(f"투자 기간: {replicated_metrics['investment_years']:.1f}년")
    
    # 5. 프로그램 기본 전략들과 비교
    comparison_results = compare_with_program_strategies(
        replicated_series, sp500_df, rsi_series, initial_capital
    )
    
    # 6. 비교 차트 생성
    create_replication_comparison_chart(replicated_series, comparison_results, initial_capital)
    
    print(f"\n=== 분석 완료 ===")
    print("수기 작업을 정확히 재현한 결과와 프로그램 전략들을 비교했습니다.")

if __name__ == "__main__":
    main()