"""
빠른 테스트 - 사용자 입력 없이 미리 정의된 전략으로 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enhanced_Backtesting_v1.0.py의 주요 함수들 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from Enhanced_Backtesting_v1.0 import *

def quick_test():
    """빠른 테스트 실행"""
    
    print("="*60)
    print("     빠른 테스트 - 미리 정의된 전략으로 실행")
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
    
    # 테스트 커스텀 전략 정의
    test_strategy = {
        '봄': 'Growth',      # 성장
        '여름': 'Momentum',  # 모멘텀  
        '가을': 'Quality',   # 퀄리티
        '겨울': 'Value'      # 가치
    }
    
    print(f"\n4. 테스트 전략 실행 중...")
    print(f"전략 구성: {test_strategy}")
    
    # 커스텀 전략과 기본 전략들 비교
    results = run_strategy_comparison(
        sp500_df, rsi_series, start_date, end_date,
        '테스트 전략', test_strategy
    )
    
    if not results:
        print("전략 비교 실패")
        return
        
    # 결과 요약
    print(f"\n{'='*80}")
    print("                     전략 비교 결과")
    print(f"{'='*80}")
    
    print(f"{'전략명':<20} {'총수익률':<12} {'CAGR':<8} {'MDD':<8} {'샤프비율':<8}")
    print("-" * 80)
    
    # 성과순으로 정렬
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['metrics']['total_return'], reverse=True)
    
    for rank, (strategy_name, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        marker = " ★" if strategy_name == '테스트 전략' else ""
        print(f"{rank}. {strategy_name:<17} "
              f"{metrics['total_return']:>10.2%} "
              f"{metrics['cagr']:>8.2%} "
              f"{metrics['mdd']:>8.2%} "
              f"{metrics['sharpe_ratio']:>8.2f}{marker}")
    
    print(f"\n테스트 완료! 실제 사용시에는 Enhanced_Backtesting_v1.0.py를 직접 실행하세요.")

if __name__ == "__main__":
    quick_test()