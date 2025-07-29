"""
수기 작업 결과와 프로그램 결과 비교 분석 도구

수기로 작성한 백테스팅 결과(11.xlsx)와 프로그램 결과를 비교하여
차이점을 찾고 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_manual_work_data(file_path):
    """
    수기 작업 데이터를 로드합니다.
    """
    try:
        # 첫 번째 행을 헤더로 사용
        df = pd.read_excel(file_path, header=1)
        
        # 날짜 컬럼을 인덱스로 설정
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"수기 작업 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
        print(f"기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        print(f"수기 작업 데이터 로딩 오류: {e}")
        return None

def analyze_manual_data_structure(df):
    """
    수기 작업 데이터 구조를 분석합니다.
    """
    print("\n=== 수기 작업 데이터 구조 분석 ===")
    print("컬럼명:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    # RSI 관련 컬럼 찾기
    rsi_cols = [col for col in df.columns if 'RSI' in str(col).upper()]
    print(f"\nRSI 관련 컬럼: {rsi_cols}")
    
    # 계절 관련 컬럼 찾기  
    season_cols = [col for col in df.columns if any(season in str(col) for season in ['봄', '여름', '가을', '겨울', '계절'])]
    print(f"계절 관련 컬럼: {season_cols}")
    
    # 포트폴리오 가치 관련 컬럼 찾기
    portfolio_cols = [col for col in df.columns if any(word in str(col) for word in ['포트폴리오', '가치', '총액', '합계'])]
    print(f"포트폴리오 관련 컬럼: {portfolio_cols}")
    
    return rsi_cols, season_cols, portfolio_cols

def extract_manual_strategy_data(df):
    """
    수기 작업에서 전략 관련 데이터를 추출합니다.
    """
    # RSI 데이터
    if 'RSI' in df.columns:
        rsi_data = df['RSI'].dropna()
        print(f"RSI 데이터 포인트: {len(rsi_data)}개")
        print(f"RSI 범위: {rsi_data.min():.1f} ~ {rsi_data.max():.1f}")
    else:
        rsi_data = None
        print("RSI 컬럼을 찾을 수 없습니다.")
    
    # 계절 분류 찾기
    season_data = None
    for col in df.columns:
        if any(season in str(col) for season in ['1월', '2월']):  # 계절 분류 컬럼으로 추정
            season_data = df[col].dropna()
            print(f"계절 데이터 컬럼: {col}")
            break
    
    # 스타일별 지수 가격 데이터 추출
    style_columns = {}
    potential_styles = ['성장', '가치', '모멘텀', '퀄리티', '로볼', '배당']
    
    for col in df.columns:
        col_str = str(col)
        for style in potential_styles:
            if style in col_str:
                style_columns[style] = col
                break
    
    print(f"발견된 스타일 컬럼: {style_columns}")
    
    return rsi_data, season_data, style_columns

def compare_with_program_results(manual_df, start_date, end_date, initial_capital=10000000):
    """
    수기 작업 결과와 프로그램 결과를 비교합니다.
    """
    print(f"\n=== 비교 분석 ({start_date} ~ {end_date}) ===")
    
    # 수기 작업에서 해당 기간 데이터 추출
    manual_period = manual_df.loc[start_date:end_date]
    print(f"수기 작업 기간 데이터: {len(manual_period)}개 포인트")
    
    # 수기 작업의 RSI 기반 계절 분류 재현
    if 'RSI' in manual_df.columns:
        rsi_manual = manual_period['RSI'].dropna()
        seasons_manual = classify_seasons_from_rsi(rsi_manual)
        
        print("\n수기 작업 계절별 분포:")
        season_dist = seasons_manual.value_counts()
        for season, count in season_dist.items():
            print(f"  {season}: {count}회 ({count/len(seasons_manual)*100:.1f}%)")
    
    # 프로그램 방식으로 동일 기간 계산
    from sp500_custom_rotation_strategy import (
        load_sp500_data, load_rsi_data, classify_market_season, 
        align_data, run_rotation_strategy, calculate_comprehensive_metrics
    )
    
    # 기본 데이터 로딩
    sp500_df = load_sp500_data('sp500_data.xlsx')
    rsi_series = load_rsi_data('RSI_DATE.xlsx')
    
    if sp500_df is not None and rsi_series is not None:
        # 프로그램 방식으로 계산
        sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
        seasons_program = classify_market_season(rsi_aligned)
        
        print("\n프로그램 계절별 분포:")
        season_dist_prog = seasons_program.value_counts()
        for season, count in season_dist_prog.items():
            print(f"  {season}: {count}회 ({count/len(seasons_program)*100:.1f}%)")
        
        # 차이점 분석
        analyze_differences(manual_period, sp500_aligned, rsi_manual if 'RSI' in manual_df.columns else None, rsi_aligned)
    
    return manual_period

def classify_seasons_from_rsi(rsi_series):
    """RSI를 기반으로 계절을 분류합니다."""
    def get_season(rsi_value):
        if pd.isna(rsi_value):
            return np.nan
        elif rsi_value >= 70:
            return '여름'
        elif rsi_value >= 50:
            return '봄'
        elif rsi_value >= 30:
            return '가을'
        else:
            return '겨울'
    
    return rsi_series.apply(get_season)

def analyze_differences(manual_data, program_data, manual_rsi, program_rsi):
    """
    수기 작업과 프로그램 결과의 차이점을 분석합니다.
    """
    print("\n=== 차이점 분석 ===")
    
    # 1. 데이터 포인트 수 비교
    print(f"데이터 포인트 수:")
    print(f"  수기 작업: {len(manual_data)}개")
    print(f"  프로그램: {len(program_data)}개")
    
    # 2. 날짜 범위 비교
    print(f"\n날짜 범위:")
    print(f"  수기 작업: {manual_data.index.min()} ~ {manual_data.index.max()}")
    print(f"  프로그램: {program_data.index.min()} ~ {program_data.index.max()}")
    
    # 3. RSI 값 비교 (있는 경우)
    if manual_rsi is not None and program_rsi is not None:
        print(f"\nRSI 비교:")
        print(f"  수기 RSI 평균: {manual_rsi.mean():.2f}")
        print(f"  프로그램 RSI 평균: {program_rsi.mean():.2f}")
        print(f"  차이: {abs(manual_rsi.mean() - program_rsi.mean()):.2f}")
        
        # 공통 날짜에서 RSI 차이 분석
        common_dates = manual_rsi.index.intersection(program_rsi.index)
        if len(common_dates) > 0:
            manual_common = manual_rsi.loc[common_dates]
            program_common = program_rsi.loc[common_dates]
            rsi_diff = abs(manual_common - program_common).mean()
            print(f"  공통 날짜 RSI 평균 차이: {rsi_diff:.2f}")
            
            # 가장 큰 차이가 나는 날짜들
            biggest_diff = abs(manual_common - program_common).nlargest(5)
            print(f"  가장 큰 차이 날짜들:")
            for date, diff in biggest_diff.items():
                print(f"    {date.strftime('%Y-%m-%d')}: {diff:.2f}")

def create_comparison_chart(manual_data, program_data):
    """
    수기 작업과 프로그램 결과 비교 차트를 생성합니다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RSI 비교 (있는 경우)
    ax1 = axes[0, 0]
    if 'RSI' in manual_data.columns:
        ax1.plot(manual_data.index, manual_data['RSI'], label='수기 작업', linewidth=2)
    ax1.set_title('RSI 비교', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RSI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 스타일별 가격 비교 (예시)
    ax2 = axes[0, 1]
    # 수기 작업에서 스타일 관련 컬럼 찾기
    style_cols = [col for col in manual_data.columns if any(style in str(col) for style in ['성장', '가치', '모멘텀'])]
    if style_cols:
        ax2.plot(manual_data.index, manual_data[style_cols[0]], label=f'수기 {style_cols[0]}')
    ax2.set_title('스타일 지수 비교', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 데이터 밀도 비교
    ax3 = axes[1, 0]
    ax3.bar(['수기 작업', '프로그램'], [len(manual_data), len(program_data) if program_data is not None else 0])
    ax3.set_title('데이터 포인트 수 비교', fontsize=12, fontweight='bold')
    ax3.set_ylabel('데이터 포인트 수')
    
    # 4. 기간 정보
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.text(0.1, 0.8, '비교 정보', fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.6, f'수기 작업 기간: {manual_data.index.min().strftime("%Y-%m")} ~ {manual_data.index.max().strftime("%Y-%m")}')
    ax4.text(0.1, 0.4, f'수기 작업 데이터: {len(manual_data)}개 포인트')
    if 'RSI' in manual_data.columns:
        ax4.text(0.1, 0.2, f'수기 RSI 범위: {manual_data["RSI"].min():.1f} ~ {manual_data["RSI"].max():.1f}')
    
    plt.tight_layout()
    plt.savefig('manual_vs_program_comparison.png', dpi=300, bbox_inches='tight')
    print("\n비교 차트가 'manual_vs_program_comparison.png'로 저장되었습니다.")

def main():
    """
    메인 함수 - 수기 작업과 프로그램 결과를 비교합니다.
    """
    print("=== 수기 작업 vs 프로그램 결과 비교 분석 ===")
    
    # 1. 수기 작업 데이터 로딩
    manual_df = load_manual_work_data('11.xlsx')
    if manual_df is None:
        return
    
    # 2. 데이터 구조 분석
    rsi_cols, season_cols, portfolio_cols = analyze_manual_data_structure(manual_df)
    
    # 3. 전략 데이터 추출
    rsi_data, season_data, style_columns = extract_manual_strategy_data(manual_df)
    
    # 4. 비교 기간 설정 (사용자가 지정할 수 있도록)
    print(f"\n수기 작업 데이터 범위: {manual_df.index.min().strftime('%Y-%m-%d')} ~ {manual_df.index.max().strftime('%Y-%m-%d')}")
    
    try:
        start_year = int(input("비교할 시작 연도 (YYYY): "))
        start_month = int(input("비교할 시작 월 (1-12): "))
        end_year = int(input("비교할 종료 연도 (YYYY): "))
        end_month = int(input("비교할 종료 월 (1-12): "))
        
        start_date = pd.Timestamp(start_year, start_month, 1)
        end_date = pd.Timestamp(end_year, end_month, 28)
        
        # 5. 비교 분석 실행
        manual_period = compare_with_program_results(manual_df, start_date, end_date)
        
        # 6. 차트 생성
        create_comparison_chart(manual_period, None)
        
        print("\n=== 분석 완료 ===")
        print("수기 작업과 프로그램 결과의 차이점을 확인했습니다.")
        print("추가로 확인하고 싶은 특정 값이나 계산 방식이 있으면 알려주세요.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()