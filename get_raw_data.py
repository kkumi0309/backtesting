"""
FinancialDataCollector를 사용하여 원시(raw) 데이터를 수집하는 예제
"""

from financial_data_collector import FinancialDataCollector

def get_raw_financial_data():
    """
    FinancialDataCollector를 사용하여 원시(raw) 금융 데이터를 수집하고
    CSV 파일로 저장하는 예제입니다.
    """
    print("=== 원시 금융 데이터 수집 시작 ===")

    # 1. 데이터 수집기 초기화 (원하는 기간으로 설정)
    # 예: 2010년 1월 1일부터 현재까지
    collector = FinancialDataCollector(start_date='2010-01-01')

    # 2. 월별 전처리 없이 원시 데이터 수집
    # collect_all_data 메서드의 monthly_processing 인자를 False로 설정합니다.
    raw_data = collector.collect_all_data(monthly_processing=False)

    if raw_data is not None:
        print("\n=== 원시 데이터 수집 완료 ===")
        print(f"데이터 형태: {raw_data.shape}")
        print(f"데이터 기간: {raw_data.index.min().strftime('%Y-%m-%d')} ~ {raw_data.index.max().strftime('%Y-%m-%d')}")
        print("\n=== 데이터 샘플 (최근 5개) ===")
        print(raw_data.tail())

        # 3. 데이터 저장
        collector.combined_data = raw_data
        collector.save_data('financial_data_raw_example.csv')

    else:
        print("✗ 데이터 수집에 실패했습니다.")

if __name__ == "__main__":
    get_raw_financial_data()