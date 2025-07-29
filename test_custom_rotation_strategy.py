"""
커스텀 로테이션 전략 테스트 스크립트
사용자 입력 없이 자동으로 커스텀 전략을 생성하여 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sp500_custom_rotation_strategy import (
    load_sp500_data, load_rsi_data, classify_market_season, 
    align_data, run_strategy_comparison, print_enhanced_results, 
    create_enhanced_comparison_chart, save_custom_strategy
)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만 하도록 설정

def test_custom_rotation_strategy():
    """
    커스텀 로테이션 전략을 자동으로 테스트합니다.
    """
    print("=== 커스텀 RSI 기반 로테이션 전략 테스트 ===\n")
    
    # 1. 데이터 로딩
    print("1. S&P 500 데이터 로딩...")
    sp500_df = load_sp500_data('sp500_data.xlsx')
    if sp500_df is None:
        return False
    
    print("\n2. RSI 데이터 로딩...")
    rsi_series = load_rsi_data('RSI_DATE.xlsx')
    if rsi_series is None:
        return False
    
    # 3. 테스트용 커스텀 전략 정의
    print("\n3. 테스트용 커스텀 전략 설정...")
    
    # 사용자가 요청한 전략: 봄-모멘텀, 여름-모멘텀, 가을-퀄리티, 겨울-로볼
    test_custom_strategy = {
        '봄': 'S&P500 Momentum',      # 상승 추세에서 모멘텀
        '여름': 'S&P500 Momentum',    # 과매수에서도 모멘텀 (위험 추구)
        '가을': 'S&P500 Quality',     # 하락 추세에서 퀄리티
        '겨울': 'S&P500 Low Volatiltiy Index'  # 과매도에서 저변동성
    }
    
    custom_strategy_name = "사용자 테스트 전략"
    
    print(f"커스텀 전략: {custom_strategy_name}")
    print("전략 구성:")
    season_icons = {'봄': '[봄]', '여름': '[여름]', '가을': '[가을]', '겨울': '[겨울]'}
    for season, style in test_custom_strategy.items():
        print(f"  {season_icons[season]} {season}: {style}")
    
    # 4. 전략 저장 테스트
    print(f"\n4. 커스텀 전략 저장 테스트...")
    save_custom_strategy(custom_strategy_name, test_custom_strategy)
    
    # 5. 테스트 파라미터 설정
    test_params = {
        'start_year': 2000,
        'start_month': 1,
        'end_year': 2015,
        'end_month': 12,
        'initial_capital': 10000000
    }
    
    print(f"\n5. 테스트 파라미터:")
    print(f"   기간: {test_params['start_year']}-{test_params['start_month']:02d} ~ {test_params['end_year']}-{test_params['end_month']:02d}")
    print(f"   초기 자본: {test_params['initial_capital']:,}원")
    
    try:
        # 6. 데이터 정렬
        print("\n6. 데이터 정렬...")
        start_date = pd.Timestamp(test_params['start_year'], test_params['start_month'], 1)
        end_date = pd.Timestamp(test_params['end_year'], test_params['end_month'], 28)
        
        sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
        
        # 7. 계절 분류
        print("\n7. 계절 분류...")
        seasons = classify_market_season(rsi_aligned)
        valid_seasons = seasons.dropna()
        
        print(f"   계절 분류 완료: {len(valid_seasons)}개 데이터 포인트")
        season_dist = valid_seasons.value_counts()
        for season, count in season_dist.items():
            print(f"     {season}: {count}회 ({count/len(valid_seasons)*100:.1f}%)")
        
        # 8. 전략 비교 실행
        print("\n8. 전략 비교 백테스팅 실행...")
        results = run_strategy_comparison(
            sp500_aligned, seasons, custom_strategy_name, 
            test_custom_strategy, test_params['initial_capital']
        )
        
        # 9. 결과 분석
        print("\n9. 결과 분석...")
        print_enhanced_results(results, custom_strategy_name)
        
        # 10. 차트 생성
        print("\n10. 차트 생성...")
        create_enhanced_comparison_chart(results, custom_strategy_name, valid_seasons)
        
        # 11. 추가 분석
        print("\n=== 추가 분석 ===")
        custom_metrics = results[custom_strategy_name]['metrics']
        
        # 다른 전략들과의 성과 비교
        other_strategies = {k: v for k, v in results.items() if k != custom_strategy_name}
        
        print(f"\n커스텀 전략 vs 기본 전략들:")
        print(f"커스텀 전략 수익률: {custom_metrics['total_return']:.2%}")
        
        for strategy_name, result in other_strategies.items():
            other_return = result['metrics']['total_return']
            diff = custom_metrics['total_return'] - other_return
            comparison = "우수" if diff > 0 else "부족"
            print(f"  vs {strategy_name}: {diff:+.2%} ({comparison})")
        
        # 리스크 조정 수익률 분석
        print(f"\n리스크 조정 성과:")
        print(f"커스텀 전략 샤프비율: {custom_metrics['sharpe_ratio']:.3f}")
        
        # 샤프 비율 순위 계산
        sharpe_rankings = [(name, result['metrics']['sharpe_ratio']) for name, result in results.items()]
        sharpe_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (strategy_name, sharpe_ratio) in enumerate(sharpe_rankings, 1):
            if strategy_name == custom_strategy_name:
                print(f"샤프비율 순위: {i}위 / {len(results)}개 전략")
                break
        
        # 계절별 효과 분석
        print(f"\n계절별 성과 분석:")
        season_performance = custom_metrics['season_performance']
        best_season = max(season_performance.items(), key=lambda x: x[1]['avg_return'])
        worst_season = min(season_performance.items(), key=lambda x: x[1]['avg_return'])
        
        print(f"최고 성과 계절: {best_season[0]} (평균 {best_season[1]['avg_return']:.2%})")
        print(f"최저 성과 계절: {worst_season[0]} (평균 {worst_season[1]['avg_return']:.2%})")
        
        print("\n[SUCCESS] 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_custom_strategies():
    """
    여러 커스텀 전략을 테스트합니다.
    """
    print("\n=== 다중 커스텀 전략 테스트 ===")
    
    # 다양한 커스텀 전략들 정의
    test_strategies = {
        "공격적 성장 전략": {
            '봄': 'S&P500 Growth',
            '여름': 'S&P500 Growth', 
            '가을': 'S&P500 Momentum',
            '겨울': 'S&P500 Growth'
        },
        "안전 우선 전략": {
            '봄': 'S&P500 Quality',
            '여름': 'S&P500 Low Volatiltiy Index',
            '가을': 'S&P500 Div Aristocrt TR Index',
            '겨울': 'S&P500 Low Volatiltiy Index'
        },
        "균형 전략": {
            '봄': 'S&P500 Growth',
            '여름': 'S&P500 Quality',
            '가을': 'S&P500 Value',
            '겨울': 'S&P500 Div Aristocrt TR Index'
        }
    }
    
    for strategy_name, strategy_rules in test_strategies.items():
        print(f"\n테스트 전략: {strategy_name}")
        for season, style in strategy_rules.items():
            print(f"  {season}: {style}")
        
        # 각 전략 저장
        save_custom_strategy(strategy_name, strategy_rules)
    
    print("\n모든 테스트 전략이 저장되었습니다.")

if __name__ == "__main__":
    success = test_custom_rotation_strategy()
    
    if success:
        # 추가 전략들도 테스트
        test_multiple_custom_strategies()
        
        print("\n[SUCCESS] 커스텀 로테이션 전략 프로그램이 정상적으로 작동합니다.")
        print("실제 사용시에는 'python sp500_custom_rotation_strategy.py' 명령어를 실행하세요.")
        print("\n사용법:")
        print("1. 프로그램 실행 후 각 계절별로 원하는 스타일 선택")
        print("2. 전략명 입력")
        print("3. 백테스팅 기간 및 초기 자본 설정")
        print("4. 기본 전략들과 성과 비교 결과 확인")
    else:
        print("\n[ERROR] 프로그램에 문제가 있습니다. 코드를 점검해주세요.")