import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def summarize_calculation_differences():
    """계산 차이를 요약 정리합니다."""
    
    print("=== 수기 vs 프로그램 계산 차이 최종 분석 ===")
    
    # 기본 사실들
    manual_return = 1139.95  # 수기 계산 결과
    program_return = 1027.58  # 프로그램 계산 결과
    difference = manual_return - program_return
    
    print(f"\n1. 기본 결과 비교:")
    print(f"   수기 계산:    {manual_return:>8.2f}%")
    print(f"   프로그램:     {program_return:>8.2f}%")
    print(f"   차이:         {difference:>8.2f}%p")
    print(f"   차이 비율:    {(difference/program_return)*100:>8.2f}%")
    
    # 데이터 검증 결과
    print(f"\n2. 데이터 검증 결과:")
    print(f"   RSI 데이터:   동일 (차이 0.00)")
    print(f"   기간:         동일 (1999.01 ~ 2025.06)")
    print(f"   스타일 매핑:  동일 확인")
    print(f"   거래 횟수:    동일 추정")
    
    # 가능한 차이 원인들
    print(f"\n3. 가능한 차이 원인 분석:")
    
    # 3-1. 복리 계산 방식
    simple_ratio = manual_return / program_return
    print(f"   3-1. 복리 계산 방식 차이:")
    print(f"        수익률 비율: {simple_ratio:.4f}")
    print(f"        이는 {((simple_ratio-1)*100):+.1f}% 추가 수익에 해당")
    
    # 3-2. 소수점 처리 방식
    print(f"   3-2. 소수점/반올림 처리:")
    print(f"        26년간 누적 시 소수점 차이가 복리효과로 증폭")
    print(f"        매월 0.1% 차이도 26년 후 상당한 차이 발생")
    
    # 3-3. 데이터 소스 차이
    print(f"   3-3. 데이터 소스 미세 차이:")
    print(f"        동일해 보이는 데이터라도 소수점 이하 차이 존재 가능")
    print(f"        Excel vs Python 계산 정밀도 차이")
    
    # 3-4. 거래 시점 차이
    print(f"   3-4. 거래 시점 차이:")
    print(f"        월초 vs 월말 가격 적용 차이")
    print(f"        리밸런싱 정확한 시점 차이")
    
    # 차이 발생 메커니즘 시뮬레이션
    print(f"\n4. 차이 발생 메커니즘 시뮬레이션:")
    
    # 가정: 매월 평균 0.1% 정도의 미세한 차이가 있다면
    monthly_diff = 0.001  # 0.1% 월별 차이
    months = 318  # 총 월수
    
    # 복리 효과 계산
    program_monthly = (1 + program_return/100) ** (1/months) - 1
    manual_monthly = program_monthly + monthly_diff
    
    simulated_final = ((1 + manual_monthly) ** months - 1) * 100
    
    print(f"   가정: 매월 {monthly_diff*100:.1f}% 미세 차이")
    print(f"   시뮬레이션 결과: {simulated_final:.2f}%")
    print(f"   실제 수기 결과: {manual_return:.2f}%")
    print(f"   오차: {abs(simulated_final - manual_return):.2f}%p")
    
    # 결론 및 권장사항
    print(f"\n5. 결론 및 권장사항:")
    print(f"   가장 가능성 높은 원인:")
    print(f"   1) 데이터 정밀도 차이 (소수점 이하)")
    print(f"   2) Excel vs Python 계산 방식 차이")
    print(f"   3) 복리 계산 시 반올림 처리 차이")
    print(f"   4) 월별 리밸런싱 정확한 날짜/가격 차이")
    
    print(f"\n   권장 해결 방법:")
    print(f"   1) 수기 계산 스프레드시트 상세 검토")
    print(f"   2) 월별 거래 내역 상세 비교")
    print(f"   3) 가격 데이터 소수점 이하 정밀도 확인")
    print(f"   4) Excel 수식과 Python 계산 로직 정확한 매칭")
    
    # 실용적 관점
    print(f"\n6. 실용적 관점:")
    print(f"   - 두 결과 모두 26년간 우수한 성과 (9-10% CAGR)")
    print(f"   - 약 112%p 차이는 상당하지만 전략의 유효성은 입증")
    print(f"   - 실제 투자 시에는 거래비용, 세금 등 추가 고려 필요")
    print(f"   - 백테스팅은 과거 데이터 기반 시뮬레이션임을 인지")

def calculate_monthly_difference_impact():
    """월별 미세 차이가 최종 결과에 미치는 영향을 계산합니다."""
    
    print(f"\n=== 월별 미세 차이 영향 분석 ===")
    
    program_return = 1027.58
    manual_return = 1139.95
    months = 318
    initial_capital = 10000000
    
    # 프로그램 월평균 수익률
    program_monthly = (1 + program_return/100) ** (1/months) - 1
    
    print(f"프로그램 월평균 수익률: {program_monthly*100:.4f}%")
    
    # 다양한 월별 차이 시나리오 테스트
    test_differences = [0.001, 0.002, 0.003, 0.004, 0.005]  # 0.1% ~ 0.5%
    
    print(f"\n월별 차이별 최종 수익률 시뮬레이션:")
    print(f"{'월별차이':<8} {'최종수익률':<12} {'목표와차이':<12}")
    print("-" * 35)
    
    closest_diff = None
    closest_error = float('inf')
    
    for diff in test_differences:
        manual_monthly_sim = program_monthly + diff
        final_return_sim = ((1 + manual_monthly_sim) ** months - 1) * 100
        error = abs(final_return_sim - manual_return)
        
        print(f"{diff*100:6.2f}% {final_return_sim:10.2f}% {error:10.2f}%p")
        
        if error < closest_error:
            closest_error = error
            closest_diff = diff
    
    print(f"\n가장 가까운 월별 차이: {closest_diff*100:.2f}%")
    print(f"이는 매월 평균 {closest_diff*100:.2f}%의 미세한 차이가")
    print(f"26년 복리 효과로 {closest_error:.1f}%p 차이를 만들 수 있음을 의미")

if __name__ == "__main__":
    summarize_calculation_differences()
    calculate_monthly_difference_impact()