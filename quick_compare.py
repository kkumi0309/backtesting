"""
수기 작업과 프로그램 결과 빠른 비교 분석
"""

import pandas as pd
import numpy as np

def quick_analysis():
    """빠른 비교 분석"""
    print("=== 수기 작업 데이터 빠른 분석 ===")
    
    # 수기 작업 데이터 로딩
    df = pd.read_excel('11.xlsx', header=1)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)
    df.sort_index(inplace=True)
    
    print(f"수기 작업 데이터 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
    print(f"총 데이터 포인트: {len(df)}개")
    
    # RSI 데이터 확인
    if 'RSI' in df.columns:
        rsi_data = df['RSI'].dropna()
        print(f"\nRSI 데이터:")
        print(f"  포인트 수: {len(rsi_data)}개")
        print(f"  범위: {rsi_data.min():.1f} ~ {rsi_data.max():.1f}")
        print(f"  평균: {rsi_data.mean():.1f}")
        
        # RSI 기반 계절 분류
        def classify_season(rsi):
            if pd.isna(rsi):
                return np.nan
            elif rsi >= 70:
                return '여름'
            elif rsi >= 50:
                return '봄'  
            elif rsi >= 30:
                return '가을'
            else:
                return '겨울'
        
        seasons = rsi_data.apply(classify_season)
        season_counts = seasons.value_counts()
        print(f"\n수기 데이터 계절별 분포:")
        for season, count in season_counts.items():
            print(f"  {season}: {count}회 ({count/len(seasons)*100:.1f}%)")
    
    # 스타일별 지수 확인
    print(f"\n사용 가능한 스타일 지수:")
    style_indices = ['성장', '가치', '모멘텀', '퀄리티', 'S&P500']
    for style in style_indices:
        matching_cols = [col for col in df.columns if style in str(col)]
        if matching_cols:
            print(f"  {style}: {matching_cols}")
    
    # 특정 기간 데이터 샘플 확인
    print(f"\n최근 5개 데이터 포인트 (주요 컬럼):")
    key_columns = ['RSI']
    # 스타일 지수 추가
    for col in df.columns:
        if any(style in str(col) for style in ['S&P500 성장', 'S&P500 가치', 'S&P500']) and col not in key_columns:
            key_columns.append(col)
            if len(key_columns) >= 6:  # 너무 많으면 5개까지만
                break
    
    available_cols = [col for col in key_columns if col in df.columns]
    if available_cols:
        print(df[available_cols].tail())
    
    # 계절별 컬럼 확인 ('1월', '2월' 컬럼)
    if '1월' in df.columns and '2월' in df.columns:
        print(f"\n계절 정보 컬럼 확인:")
        print(f"1월 컬럼 고유값: {df['1월'].value_counts()}")
        print(f"2월 컬럼 고유값: {df['2월'].value_counts()}")

if __name__ == "__main__":
    quick_analysis()