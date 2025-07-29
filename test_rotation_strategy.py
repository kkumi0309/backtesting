"""
RSI 기반 로테이션 전략 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sp500_rotation_strategy import (
    load_data, calculate_rsi, classify_market_season, 
    run_all_strategies, print_detailed_results, create_comparison_chart
)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만 하도록 설정

def test_rotation_strategy():
    """
    RSI 기반 로테이션 전략을 자동으로 테스트합니다.
    """
    print("=== RSI 기반 로테이션 전략 테스트 ===\n")
    
    # 1. 데이터 로딩
    print("1. 데이터 로딩...")
    df = load_data('sp500_data.xlsx')
    if df is None:
        return False
    
    # 2. 테스트 파라미터 설정
    test_params = {
        'start_year': 2020,
        'start_month': 1,
        'end_year': 2023,
        'end_month': 12,
        'initial_capital': 10000000,
        'rsi_period': 14
    }
    
    print(f"\n2. 테스트 파라미터:")
    print(f"   기간: {test_params['start_year']}-{test_params['start_month']:02d} ~ {test_params['end_year']}-{test_params['end_month']:02d}")
    print(f"   초기 자본: {test_params['initial_capital']:,}원")
    print(f"   RSI 기간: {test_params['rsi_period']}일")
    
    try:
        # 3. 데이터 필터링
        print("\n3. 데이터 필터링...")
        start_date = pd.Timestamp(test_params['start_year'], test_params['start_month'], 1)
        end_date = pd.Timestamp(test_params['end_year'], test_params['end_month'], 28)
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df.loc[mask].copy()
        
        print(f"   필터링된 데이터: {len(filtered_df)}개 데이터 포인트")
        
        # 4. RSI 계산
        print("\n4. RSI 계산...")
        sp500_prices = filtered_df['S&P500 Growth']
        rsi = calculate_rsi(sp500_prices, test_params['rsi_period'])
        print(f"   RSI 계산 완료: 평균 RSI = {rsi.mean():.1f}")
        
        # 5. 계절 분류
        print("\n5. 계절 분류...")
        seasons = classify_market_season(rsi)
        valid_seasons = seasons.dropna()
        
        print(f"   계절 분류 완료: {len(valid_seasons)}개 데이터 포인트")
        season_dist = valid_seasons.value_counts()
        for season, count in season_dist.items():
            print(f"     {season}: {count}회 ({count/len(valid_seasons)*100:.1f}%)")
        
        # 6. 로테이션 전략 실행
        print("\n6. 로테이션 전략 실행...")
        results = run_all_strategies(filtered_df, seasons, test_params['initial_capital'])
        
        # 7. 결과 분석
        print("\n7. 결과 분석...")
        print_detailed_results(results)
        
        # 8. 차트 생성
        print("\n8. 차트 생성...")
        create_comparison_chart(results, filtered_df, valid_seasons)
        
        # 9. 추가 통계
        print("\n=== 추가 통계 ===")
        best_strategy = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
        worst_strategy = min(results.items(), key=lambda x: x[1]['metrics']['total_return'])
        
        print(f"최고 성과 전략: {best_strategy[0]} (+{best_strategy[1]['metrics']['total_return']:.2%})")
        print(f"최저 성과 전략: {worst_strategy[0]} (+{worst_strategy[1]['metrics']['total_return']:.2%})")
        
        performance_gap = best_strategy[1]['metrics']['total_return'] - worst_strategy[1]['metrics']['total_return']
        print(f"성과 격차: {performance_gap:.2%}")
        
        # 거래 횟수 분석
        print(f"\n=== 거래 빈도 분석 ===")
        for strategy_name, result in results.items():
            trade_count = len([t for t in result['transactions'] if t['action'] in ['매수', '매도']])
            print(f"{strategy_name}: {trade_count}회 거래")
        
        print("\n[SUCCESS] 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rotation_strategy()
    if success:
        print("\n[SUCCESS] 프로그램이 정상적으로 작동합니다.")
        print("실제 사용시에는 'python sp500_rotation_strategy.py' 명령어를 실행하세요.")
    else:
        print("\n[ERROR] 프로그램에 문제가 있습니다. 코드를 점검해주세요.")