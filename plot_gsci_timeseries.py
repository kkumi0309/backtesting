"""
수정된 FinancialDataCollector를 사용하여 데이터를 재수집하고,
S&P GSCI 원자재 지수의 시계열 그래프를 생성하는 스크립트입니다.
"""

import matplotlib.pyplot as plt
from financial_data_collector import FinancialDataCollector

# 한글 폰트 설정 (Windows의 'Malgun Gothic' 기준)
# 사용자의 환경에 맞는 폰트로 변경해야 할 수 있습니다.
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Malgun Gothic 폰트를 찾을 수 없습니다. 기본 폰트로 그래프를 생성합니다.")

def recollect_and_plot_gsci():
    """
    데이터를 다시 수집하고 S&P GSCI 지수 시계열 그래프를 그립니다.
    """
    print("=== 데이터 재수집 및 원자재 지수 시각화 시작 ===")

    # 1. 데이터 수집기 초기화 (2010년부터 현재까지)
    # 이전 대화에서 수정한 S&P GSCI 수집 로직이 사용됩니다.
    collector = FinancialDataCollector(start_date='2010-01-01')

    # 2. 월별 전처리 없이 원시(raw) 데이터 수집
    # 이 과정에서 'SP_GSCI'가 '원자재지수'로 이름이 변경됩니다.
    raw_data = collector.collect_all_data(monthly_processing=False)

    if raw_data is not None and '원자재지수' in raw_data.columns:
        print("\n✓ 데이터 수집 및 '원자재지수' 확인 완료.")
        print("데이터 샘플 (최근 5개):")
        print(raw_data[['원자재지수']].tail())

        # 3. 시계열 그래프 생성
        print("\n그래프 생성 중...")
        plt.figure(figsize=(14, 7))
        plt.plot(raw_data.index, raw_data['원자재지수'], label='S&P GSCI', color='darkorange', linewidth=1.5)
        plt.title('S&P GSCI 원자재 지수 시계열 그래프 (2010-현재)', fontsize=16, pad=20)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('지수 (Index)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 4. 그래프 저장 및 출력
        filepath = "C:\\Users\\wnghk\\Desktop\\ACADEMY\\2025-1.5\\계량경제\\sp_gsci_timeseries.png"
        plt.savefig(filepath, dpi=300)
        print(f"✓ 그래프가 '{filepath}'에 저장되었습니다.")
        plt.show()
    elif raw_data is None:
        print("✗ 데이터 수집에 실패했습니다.")
    else:
        print("✗ 수집된 데이터에 '원자재지수' 컬럼이 없습니다. 데이터 수집 과정을 확인해주세요.")

if __name__ == "__main__":
    recollect_and_plot_gsci()