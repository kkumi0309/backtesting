"""
월별 매수/매도 기능이 포함된 백테스팅 프로그램 테스트 스크립트
사용자 입력 없이 자동으로 백테스팅을 실행하여 프로그램이 정상 작동하는지 확인합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sp500_backtesting_with_trading import (
    load_data, filter_data_by_period, run_backtesting_with_trading, 
    calculate_performance_metrics, print_results, create_and_save_chart
)

def test_trading_backtesting():
    """
    월별 매수/매도 백테스팅 프로그램을 자동으로 테스트합니다.
    """
    print("=== S&P 500 백테스팅 프로그램 (매수/매도 기능) 테스트 ===\n")
    
    # 1. 데이터 로딩
    print("1. 데이터 로딩...")
    df = load_data('sp500_data.xlsx')
    
    if df is None:
        print("데이터 로딩 실패!")
        return False
    
    # 2. 테스트 파라미터 설정
    print("\n2. 테스트 파라미터 설정...")
    test_params = {
        'start_year': 2020,
        'start_month': 1,
        'end_year': 2022,
        'end_month': 12,
        'selected_index': 'S&P500 Growth',
        'initial_capital': 10000000  # 1천만원
    }
    
    # 3. 테스트 거래 액션 설정
    test_trading_actions = {
        '2020-01': 3000000,   # 300만원 매수
        '2020-02': 0,         # 보유
        '2020-03': 2000000,   # 200만원 매수 (코로나 하락 시점)
        '2020-04': 0,         # 보유
        '2020-05': 0,         # 보유
        '2020-06': 1000000,   # 100만원 매수
        '2020-07': 0,         # 보유
        '2020-08': 0,         # 보유
        '2020-09': -1000000,  # 100만원 매도
        '2020-10': 0,         # 보유
        '2020-11': 0,         # 보유
        '2020-12': 1500000,   # 150만원 매수
        '2021-01': 0,         # 보유
        '2021-02': 0,         # 보유
        '2021-03': -2000000,  # 200만원 매도
        '2021-04': 0,         # 보유
        '2021-05': 0,         # 보유
        '2021-06': 1000000,   # 100만원 매수
        '2021-07': 0,         # 보유
        '2021-08': 0,         # 보유
        '2021-09': 0,         # 보유
        '2021-10': 0,         # 보유
        '2021-11': 0,         # 보유
        '2021-12': -1500000,  # 150만원 매도
        '2022-01': 0,         # 보유
        '2022-02': 0,         # 보유
        '2022-03': 2000000,   # 200만원 매수
        '2022-04': 0,         # 보유
        '2022-05': 0,         # 보유
        '2022-06': 0,         # 보유
        '2022-07': 0,         # 보유
        '2022-08': 0,         # 보유
        '2022-09': 0,         # 보유
        '2022-10': 1000000,   # 100만원 매수 (하락 시점)
        '2022-11': 0,         # 보유
        '2022-12': 0,         # 보유
    }
    
    print(f"테스트 설정:")
    print(f"- 기간: {test_params['start_year']}-{test_params['start_month']:02d} ~ {test_params['end_year']}-{test_params['end_month']:02d}")
    print(f"- 선택 지수: {test_params['selected_index']}")
    print(f"- 초기 투자 원금: {test_params['initial_capital']:,}원")
    print(f"- 거래 설정: {len([k for k, v in test_trading_actions.items() if v != 0])}번의 매수/매도")
    
    try:
        # 3. 데이터 필터링
        print("\n3. 데이터 필터링...")
        filtered_df = filter_data_by_period(
            df, 
            test_params['start_year'], 
            test_params['start_month'],
            test_params['end_year'], 
            test_params['end_month']
        )
        
        # 4. 백테스팅 실행
        print("\n4. 월별 매수/매도 백테스팅 실행...")
        portfolio_series, transactions = run_backtesting_with_trading(
            filtered_df, 
            test_params['selected_index'], 
            test_params['initial_capital'],
            test_trading_actions
        )
        
        print(f"백테스팅 완료: {len(portfolio_series)}개 데이터 포인트, {len(transactions)}건 거래 내역")
        
        # 5. 성과 지표 계산
        print("\n5. 성과 지표 계산...")
        metrics = calculate_performance_metrics(portfolio_series, test_params['initial_capital'])
        
        # 6. 결과 출력
        print_results(metrics, test_params['selected_index'], test_params['initial_capital'], transactions)
        
        # 7. 차트 생성 (저장만 하고 표시하지 않음)
        print("\n7. 차트 생성...")
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 저장만 하도록 설정
        
        create_and_save_chart(portfolio_series, test_params['selected_index'], test_params['initial_capital'], transactions)
        
        print("\n[SUCCESS] 모든 테스트가 성공적으로 완료되었습니다!")
        
        # 8. 추가 분석 정보 출력
        print("\n=== 추가 분석 정보 ===")
        total_buy = sum([amount for amount in test_trading_actions.values() if amount > 0])
        total_sell = abs(sum([amount for amount in test_trading_actions.values() if amount < 0]))
        print(f"총 매수 금액: {total_buy:,}원")
        print(f"총 매도 금액: {total_sell:,}원")
        print(f"순 투자 금액: {test_params['initial_capital'] + total_buy - total_sell:,}원")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trading_backtesting()
    if success:
        print("\n프로그램이 정상적으로 작동합니다.")
        print("실제 사용시에는 'python sp500_backtesting_with_trading.py' 명령어를 실행하세요.")
    else:
        print("\n프로그램에 문제가 있습니다. 코드를 점검해주세요.")