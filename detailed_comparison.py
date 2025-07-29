"""
수기 작업과 프로그램 결과 상세 비교 분석
RSI_DATE.xlsx 파일을 기준으로 정확한 비교를 수행합니다.
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

def load_rsi_reference_data():
    """RSI_DATE.xlsx 파일을 로드합니다."""
    try:
        df = pd.read_excel('RSI_DATE.xlsx', skiprows=1)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        rsi_series = df['RSI'].dropna()
        print(f"RSI_DATE.xlsx 로딩 완료: {len(rsi_series)}개 데이터 포인트")
        print(f"RSI 기간: {rsi_series.index.min().strftime('%Y-%m-%d')} ~ {rsi_series.index.max().strftime('%Y-%m-%d')}")
        print(f"RSI 범위: {rsi_series.min():.1f} ~ {rsi_series.max():.1f}")
        print(f"RSI 평균: {rsi_series.mean():.1f}")
        
        return rsi_series
    except Exception as e:
        print(f"RSI_DATE.xlsx 로딩 오류: {e}")
        return None

def load_manual_work_data():
    """수기 작업 데이터(11.xlsx)를 로드합니다."""
    try:
        df = pd.read_excel('11.xlsx', header=1)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"수기 작업 데이터 로딩 완료: {len(df)}개 행")
        print(f"수기 작업 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    except Exception as e:
        print(f"수기 작업 데이터 로딩 오류: {e}")
        return None

def compare_rsi_data(rsi_reference, manual_df):
    """RSI 데이터를 비교합니다."""
    print("\n=== RSI 데이터 비교 ===")
    
    # 수기 작업의 RSI
    if 'RSI' in manual_df.columns:
        rsi_manual = manual_df['RSI'].dropna()
        print(f"수기 작업 RSI: {len(rsi_manual)}개 포인트, 평균 {rsi_manual.mean():.1f}")
    else:
        print("수기 작업에 RSI 컬럼이 없습니다.")
        return None, None
    
    print(f"RSI_DATE.xlsx: {len(rsi_reference)}개 포인트, 평균 {rsi_reference.mean():.1f}")
    
    # 공통 날짜 찾기
    common_dates = rsi_reference.index.intersection(rsi_manual.index)
    print(f"공통 날짜: {len(common_dates)}개")
    
    if len(common_dates) > 0:
        ref_common = rsi_reference.loc[common_dates]
        manual_common = rsi_manual.loc[common_dates]
        
        # RSI 차이 분석
        rsi_diff = abs(ref_common - manual_common)
        print(f"RSI 평균 차이: {rsi_diff.mean():.3f}")
        print(f"RSI 최대 차이: {rsi_diff.max():.3f}")
        
        # 완전히 일치하는지 확인
        identical = (ref_common == manual_common).all()
        print(f"RSI 데이터 일치 여부: {identical}")
        
        if not identical:
            print("\n가장 큰 차이가 나는 날짜들:")
            biggest_diff = rsi_diff.nlargest(5)
            for date, diff in biggest_diff.items():
                print(f"  {date.strftime('%Y-%m-%d')}: 차이 {diff:.3f} (기준: {ref_common[date]:.1f}, 수기: {manual_common[date]:.1f})")
        
        return ref_common, manual_common
    
    return None, None

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

def decode_manual_strategy(manual_df):
    """수기 작업의 숫자 코딩 전략을 해독합니다."""
    print("\n=== 수기 작업 전략 해독 ===")
    
    # 스타일 관련 컬럼들 찾기
    strategy_columns = []
    for col in manual_df.columns:
        if 'S&P500' in str(col) and any(x in str(col) for x in ['성장', '가치']) == False:
            # 순수한 S&P500 컬럼들 (성장, 가치 등이 붙지 않은)
            unique_vals = manual_df[col].dropna().unique()
            if len(unique_vals) > 1 and all(isinstance(x, (int, float)) for x in unique_vals):
                strategy_columns.append(col)
    
    print(f"전략 관련 컬럼들: {strategy_columns}")
    
    # 각 컬럼의 고유값 분석
    style_mapping = {
        1: 'S&P500 Growth',
        2: 'S&P500 Value', 
        3: 'S&P500 Momentum',
        4: 'S&P500 Quality',
        5: 'S&P500 Low Volatiltiy Index',
        6: 'S&P500 Div Aristocrt TR Index'
    }
    
    for col in strategy_columns:
        unique_vals = sorted(manual_df[col].dropna().unique())
        print(f"\n{col} 컬럼:")
        print(f"  고유값: {unique_vals}")
        print(f"  추정 매핑:")
        for val in unique_vals:
            if val in style_mapping:
                count = (manual_df[col] == val).sum()
                print(f"    {val} → {style_mapping[val]} ({count}회)")
    
    return strategy_columns, style_mapping

def analyze_strategy_transitions(manual_df, rsi_data, strategy_columns):
    """전략 전환 시점을 분석합니다."""
    print("\n=== 전략 전환 분석 ===")
    
    if not strategy_columns or 'RSI' not in manual_df.columns:
        print("전략 분석에 필요한 데이터가 부족합니다.")
        return
    
    # 첫 번째 전략 컬럼 사용
    strategy_col = strategy_columns[0]
    strategy_data = manual_df[strategy_col].dropna()
    rsi_manual = manual_df['RSI'].dropna()
    
    # 공통 날짜
    common_dates = strategy_data.index.intersection(rsi_manual.index)
    if len(common_dates) == 0:
        print("전략과 RSI 데이터의 공통 날짜가 없습니다.")
        return
    
    strategy_common = strategy_data.loc[common_dates]
    rsi_common = rsi_manual.loc[common_dates]
    
    # 계절별 전략 사용 패턴 분석
    seasons = classify_seasons_from_rsi(rsi_common)
    
    print(f"\n계절별 전략 사용 패턴 ({strategy_col}):")
    style_mapping = {
        1: 'Growth', 2: 'Value', 3: 'Momentum', 
        4: 'Quality', 5: 'Low Vol', 6: 'Dividend'
    }
    
    for season in ['봄', '여름', '가을', '겨울']:
        season_mask = seasons == season
        if season_mask.sum() > 0:
            season_strategies = strategy_common[season_mask]
            strategy_counts = season_strategies.value_counts()
            print(f"\n  {season} ({season_mask.sum()}회):")
            for strategy_num, count in strategy_counts.items():
                style_name = style_mapping.get(strategy_num, f'스타일{strategy_num}')
                print(f"    {style_name}: {count}회 ({count/season_mask.sum()*100:.1f}%)")

def compare_with_program_logic(rsi_reference, manual_df, start_year=2000, end_year=2015):
    """프로그램 로직과 비교합니다."""
    print(f"\n=== 프로그램 로직과 비교 ({start_year}-{end_year}) ===")
    
    # 기간 필터링
    start_date = pd.Timestamp(start_year, 1, 1)
    end_date = pd.Timestamp(end_year, 12, 31)
    
    rsi_period = rsi_reference.loc[start_date:end_date]
    manual_period = manual_df.loc[start_date:end_date]
    
    print(f"비교 기간 데이터:")
    print(f"  RSI_DATE.xlsx: {len(rsi_period)}개 포인트")
    print(f"  수기 작업: {len(manual_period)}개 포인트")
    
    # 계절 분류 비교
    if len(rsi_period) > 0:
        seasons_reference = classify_seasons_from_rsi(rsi_period)
        season_dist_ref = seasons_reference.value_counts()
        
        print(f"\nRSI_DATE.xlsx 기반 계절 분포:")
        for season, count in season_dist_ref.items():
            print(f"  {season}: {count}회 ({count/len(seasons_reference)*100:.1f}%)")
    
    if 'RSI' in manual_df.columns and len(manual_period) > 0:
        manual_rsi_period = manual_period['RSI'].dropna()
        if len(manual_rsi_period) > 0:
            seasons_manual = classify_seasons_from_rsi(manual_rsi_period)
            season_dist_manual = seasons_manual.value_counts()
            
            print(f"\n수기 작업 기반 계절 분포:")
            for season, count in season_dist_manual.items():
                print(f"  {season}: {count}회 ({count/len(seasons_manual)*100:.1f}%)")

def create_detailed_comparison_chart(rsi_reference, manual_df):
    """상세 비교 차트를 생성합니다."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RSI 비교
    ax1 = axes[0, 0]
    
    # 공통 날짜 찾기
    if 'RSI' in manual_df.columns:
        manual_rsi = manual_df['RSI'].dropna()
        common_dates = rsi_reference.index.intersection(manual_rsi.index)
        
        if len(common_dates) > 20:  # 충분한 데이터가 있는 경우
            # 최근 1년 데이터만 플롯
            recent_dates = common_dates[-12:]
            ax1.plot(recent_dates, rsi_reference.loc[recent_dates], 
                    label='RSI_DATE.xlsx', linewidth=2, marker='o')
            ax1.plot(recent_dates, manual_rsi.loc[recent_dates], 
                    label='수기 작업', linewidth=2, marker='s')
    
    ax1.set_title('RSI 비교 (최근 12개월)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RSI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 계절별 분포 비교
    ax2 = axes[0, 1]
    if 'RSI' in manual_df.columns:
        manual_rsi = manual_df['RSI'].dropna()
        
        seasons_ref = classify_seasons_from_rsi(rsi_reference)
        seasons_manual = classify_seasons_from_rsi(manual_rsi)
        
        season_counts_ref = seasons_ref.value_counts()
        season_counts_manual = seasons_manual.value_counts()
        
        seasons = ['봄', '여름', '가을', '겨울']
        ref_values = [season_counts_ref.get(s, 0) for s in seasons]
        manual_values = [season_counts_manual.get(s, 0) for s in seasons]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ax2.bar(x - width/2, ref_values, width, label='RSI_DATE.xlsx', alpha=0.8)
        ax2.bar(x + width/2, manual_values, width, label='수기 작업', alpha=0.8)
        ax2.set_title('계절별 분포 비교', fontsize=12, fontweight='bold')
        ax2.set_ylabel('빈도')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seasons)
        ax2.legend()
    
    # 3. RSI 차이 히스토그램
    ax3 = axes[1, 0]
    if 'RSI' in manual_df.columns:
        manual_rsi = manual_df['RSI'].dropna()
        common_dates = rsi_reference.index.intersection(manual_rsi.index)
        
        if len(common_dates) > 0:
            rsi_diff = rsi_reference.loc[common_dates] - manual_rsi.loc[common_dates]
            ax3.hist(rsi_diff, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_title(f'RSI 차이 분포 (평균: {rsi_diff.mean():.3f})', fontsize=12, fontweight='bold')
            ax3.set_xlabel('RSI 차이 (RSI_DATE - 수기)')
            ax3.set_ylabel('빈도')
            ax3.grid(True, alpha=0.3)
    
    # 4. 통계 요약
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 통계 정보 텍스트
    stats_text = "=== 비교 통계 ===\n\n"
    stats_text += f"RSI_DATE.xlsx:\n"
    stats_text += f"  데이터 포인트: {len(rsi_reference)}개\n"
    stats_text += f"  기간: {rsi_reference.index.min().strftime('%Y-%m')} ~ {rsi_reference.index.max().strftime('%Y-%m')}\n"
    stats_text += f"  평균 RSI: {rsi_reference.mean():.1f}\n\n"
    
    if 'RSI' in manual_df.columns:
        manual_rsi = manual_df['RSI'].dropna()
        stats_text += f"수기 작업:\n"
        stats_text += f"  데이터 포인트: {len(manual_rsi)}개\n"
        stats_text += f"  기간: {manual_rsi.index.min().strftime('%Y-%m')} ~ {manual_rsi.index.max().strftime('%Y-%m')}\n"
        stats_text += f"  평균 RSI: {manual_rsi.mean():.1f}\n\n"
        
        common_dates = rsi_reference.index.intersection(manual_rsi.index)
        stats_text += f"공통 데이터:\n"
        stats_text += f"  공통 날짜: {len(common_dates)}개\n"
        if len(common_dates) > 0:
            rsi_diff = abs(rsi_reference.loc[common_dates] - manual_rsi.loc[common_dates])
            stats_text += f"  평균 차이: {rsi_diff.mean():.3f}\n"
            stats_text += f"  최대 차이: {rsi_diff.max():.3f}"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('detailed_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("\n상세 비교 차트가 'detailed_comparison_analysis.png'로 저장되었습니다.")

def main():
    """메인 함수"""
    print("=== 수기 작업 vs 프로그램 상세 비교 분석 ===")
    
    # 1. 데이터 로딩
    rsi_reference = load_rsi_reference_data()
    manual_df = load_manual_work_data()
    
    if rsi_reference is None or manual_df is None:
        print("필요한 데이터를 로딩할 수 없습니다.")
        return
    
    # 2. RSI 데이터 비교
    ref_common, manual_common = compare_rsi_data(rsi_reference, manual_df)
    
    # 3. 수기 작업 전략 해독
    strategy_columns, style_mapping = decode_manual_strategy(manual_df)
    
    # 4. 전략 전환 분석
    analyze_strategy_transitions(manual_df, rsi_reference, strategy_columns)
    
    # 5. 프로그램 로직과 비교
    compare_with_program_logic(rsi_reference, manual_df)
    
    # 6. 상세 비교 차트 생성
    create_detailed_comparison_chart(rsi_reference, manual_df)
    
    print("\n=== 분석 완료 ===")
    print("RSI_DATE.xlsx를 기준으로 한 상세 비교가 완료되었습니다.")

if __name__ == "__main__":
    main()