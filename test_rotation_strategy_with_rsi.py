"""
기존 RSI 데이터를 활용한 로테이션 전략 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sp500_rotation_strategy_with_rsi import (
    load_sp500_data, load_rsi_data, classify_market_season, 
    align_data, run_all_strategies, print_detailed_results, create_comparison_chart
)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만 하도록 설정

def test_rotation_strategy_with_rsi():
    """
    기존 RSI 데이터를 활용한 로테이션 전략을 자동으로 테스트합니다.
    """
    print("=== RSI 기반 로테이션 전략 테스트 (기존 RSI 데이터 활용) ===\n")
    
    # 1. 데이터 로딩
    print("1. S&P 500 데이터 로딩...")
    sp500_df = load_sp500_data('sp500_data.xlsx')
    if sp500_df is None:
        return False
    
    print("\n2. RSI 데이터 로딩...")
    rsi_series = load_rsi_data('RSI_DATE.xlsx')
    if rsi_series is None:
        return False
    
    # 3. 테스트 파라미터 설정 (RSI 데이터 범위에 맞춤: 1999-2025)
    test_params = {
        'start_year': 1999,
        'start_month': 1,
        'end_year': 2010,
        'end_month': 12,
        'initial_capital': 10000000
    }
    
    print(f"\n3. 테스트 파라미터:")
    print(f"   기간: {test_params['start_year']}-{test_params['start_month']:02d} ~ {test_params['end_year']}-{test_params['end_month']:02d}")
    print(f"   초기 자본: {test_params['initial_capital']:,}원")
    
    try:
        # 4. 데이터 정렬
        print("\n4. 데이터 정렬...")
        start_date = pd.Timestamp(test_params['start_year'], test_params['start_month'], 1)
        end_date = pd.Timestamp(test_params['end_year'], test_params['end_month'], 28)
        
        sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
        
        # 5. 계절 분류
        print("\n5. 계절 분류...")
        seasons = classify_market_season(rsi_aligned)
        valid_seasons = seasons.dropna()
        
        print(f"   계절 분류 완료: {len(valid_seasons)}개 데이터 포인트")
        season_dist = valid_seasons.value_counts()
        for season, count in season_dist.items():
            print(f"     {season}: {count}회 ({count/len(valid_seasons)*100:.1f}%)")
        
        # 6. 로테이션 전략 실행
        print("\n6. 로테이션 전략 실행...")
        results = run_all_strategies(sp500_aligned, seasons, test_params['initial_capital'])
        
        # 7. 결과 분석
        print("\n7. 결과 분석...")
        print_detailed_results(results)
        
        # 8. 차트 생성
        print("\n8. 차트 생성...")
        create_comparison_chart(results, sp500_aligned, valid_seasons)
        
        # 9. 추가 통계
        print("\n=== 추가 통계 ===")
        best_strategy = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
        worst_strategy = min(results.items(), key=lambda x: x[1]['metrics']['total_return'])
        
        print(f"최고 성과 전략: {best_strategy[0]} (+{best_strategy[1]['metrics']['total_return']:.2%})")
        print(f"최저 성과 전략: {worst_strategy[0]} (+{worst_strategy[1]['metrics']['total_return']:.2%})")
        
        performance_gap = best_strategy[1]['metrics']['total_return'] - worst_strategy[1]['metrics']['total_return']
        print(f"성과 격차: {performance_gap:.2%}")
        
        # RSI 분포 분석
        print(f"\n=== RSI 분포 분석 ===")
        print(f"평균 RSI: {rsi_aligned.mean():.1f}")
        print(f"RSI 범위: {rsi_aligned.min():.1f} ~ {rsi_aligned.max():.1f}")
        print(f"과매수 비율 (RSI≥70): {(rsi_aligned >= 70).sum() / len(rsi_aligned) * 100:.1f}%")
        print(f"과매도 비율 (RSI≤30): {(rsi_aligned <= 30).sum() / len(rsi_aligned) * 100:.1f}%")
        
        print("\n[SUCCESS] 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rotation_strategy_with_rsi()
    if success:
        print("\n[SUCCESS] 프로그램이 정상적으로 작동합니다.")
        print("실제 사용시에는 'python sp500_rotation_strategy_with_rsi.py' 명령어를 실행하세요.")
    else:
        print("\n[ERROR] 프로그램에 문제가 있습니다. 코드를 점검해주세요.")