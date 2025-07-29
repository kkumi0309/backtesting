"""
1990년 1월 1일부터 현재까지 금융 데이터 분석
"""

from financial_data_collector import FinancialDataCollector
import pandas as pd

def main():
    """1990-현재 기간 금융 데이터 수집 및 분석"""
    print("=== 1990년-현재 금융 데이터 분석 ===")
    print("분석 기간: 1990년 1월 1일 ~ 현재")
    
    # 데이터 수집기 초기화 (1990년부터 현재까지)
    collector = FinancialDataCollector(start_date='1990-01-01')
    
    # 월별 전처리된 데이터 수집
    monthly_data = collector.collect_all_data(monthly_processing=True)
    
    if monthly_data is not None:
        print("\n=== 최종 데이터셋 정보 ===")
        print(f"데이터 형태: {monthly_data.shape}")
        print(f"분석 기간: {monthly_data.index.min().strftime('%Y-%m')} ~ {monthly_data.index.max().strftime('%Y-%m')}")
        print(f"총 기간: {len(monthly_data)} 개월")
        print(f"변수: {list(monthly_data.columns)}")
        
        print("\n=== 요약 통계 ===")
        print(monthly_data.describe())
        
        print("\n=== 최근 12개월 데이터 ===")
        print(monthly_data.tail(12))
        
        # 데이터 저장
        collector.combined_data = monthly_data
        collector.save_data('financial_data_1990_current.csv')
        
        print(f"\n✓ 데이터가 'financial_data_1990_current.csv'에 저장되었습니다.")
        
        return monthly_data
    else:
        print("✗ 데이터 수집에 실패했습니다.")
        return None

if __name__ == "__main__":
    data = main()
    