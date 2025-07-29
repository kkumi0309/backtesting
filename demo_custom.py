"""
커스텀 전략 데모 - 사용자 입력 없이 자동으로 커스텀 전략을 시연
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import seaborn as sns
import os

# Enhanced_Backtesting_v1.0.py의 주요 함수들을 임포트
exec(open('Enhanced_Backtesting_v1.0.py').read())

def demo_custom_strategy():
    """커스텀 전략 데모 실행"""
    
    print("="*60)
    print("     커스텀 전략 데모 - 자동 실행")
    print("="*60)
    
    # 현재 스크립트 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터 로딩
    print("\n1. 데이터 로딩 중...")
    sp500_file = os.path.join(script_dir, 'sp500_data.xlsx')
    rsi_file = os.path.join(script_dir, 'RSI_DATE.xlsx')
    
    sp500_df = load_sp500_data(sp500_file)
    rsi_series = load_rsi_data(rsi_file)
    
    if sp500_df is None or rsi_series is None:
        print("데이터 로딩 실패")
        return
    
    # 백테스팅 기간
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2025-06-30')
    
    print(f"\n2. 백테스팅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    # 단순 보유 수익률 계산
    print(f"\n3. 단순 보유 수익률 계산 중...")
    buy_hold_results = calculate_buy_and_hold_returns(sp500_df, start_date, end_date)
    
    # 데모 커스텀 전략들 정의
    demo_strategies = {
        '공격적 성장 전략': {
            '봄': 'Growth',      # 성장
            '여름': 'Momentum',  # 모멘텀  
            '가을': 'Quality',   # 퀄리티
            '겨울': 'Value'      # 가치
        },
        '안전 중심 전략': {
            '봄': 'Quality',     # 퀄리티
            '여름': 'Low Vol',   # 로우볼
            '가을': 'Dividend',  # 배당
            '겨울': 'Value'      # 가치
        },
        '균형 잡힌 전략': {
            '봄': 'Quality',     # 퀄리티
            '여름': 'Growth',    # 성장
            '가을': 'Value',     # 가치
            '겨울': 'Dividend'   # 배당
        }
    }
    
    print(f"\n4. 커스텀 전략 비교 실행 중...")
    print(f"테스트할 전략 수: {len(demo_strategies)}개")
    
    all_results = {}
    
    # 각 데모 전략 실행
    for strategy_name, rules in demo_strategies.items():
        print(f"\n--- {strategy_name} 실행 중 ---")
        print(f"전략 구성: {rules}")
        
        result = run_dynamic_strategy(sp500_df, rsi_series, start_date, end_date, 
                                    initial_capital=10000000, strategy_rules=rules)
        
        all_results[strategy_name] = result
        
        metrics = result['metrics']
        print(f"결과: 총수익률 {metrics['total_return']:.2%}, CAGR {metrics['cagr']:.2%}")
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("                     커스텀 전략 비교 결과")
    print(f"{'='*80}")
    
    print(f"{'전략명':<20} {'총수익률':<12} {'CAGR':<8} {'MDD':<8} {'샤프비율':<8}")
    print("-" * 80)
    
    # 성과순으로 정렬
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['metrics']['total_return'], reverse=True)
    
    for rank, (strategy_name, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        print(f"{rank}. {strategy_name:<17} "
              f"{metrics['total_return']:>10.2%} "
              f"{metrics['cagr']:>8.2%} "
              f"{metrics['mdd']:>8.2%} "
              f"{metrics['sharpe_ratio']:>8.2f}")
    
    # 최고 단순보유와 비교
    best_buy_hold = max(buy_hold_results.items(), key=lambda x: x[1]['total_return'])
    best_custom = sorted_results[0]
    
    print(f"\n{'='*50}")
    print("         단순보유 vs 최적 커스텀 전략")
    print(f"{'='*50}")
    print(f"최고 단순보유: {best_buy_hold[0].replace('S&P500 ', '')}")
    print(f"  - 총수익률: {best_buy_hold[1]['total_return']:.2%}")
    print(f"최적 커스텀전략: {best_custom[0]}")
    print(f"  - 총수익률: {best_custom[1]['metrics']['total_return']:.2%}")
    
    outperformance = best_custom[1]['metrics']['total_return'] - best_buy_hold[1]['total_return']
    print(f"초과/부족 수익률: {outperformance:+.2%}")
    
    print(f"\n데모 완료! 실제로는 사용자가 직접 각 계절별 스타일을 선택할 수 있습니다.")

if __name__ == "__main__":
    demo_custom_strategy()