import pandas as pd
import numpy as np

def analyze_csv_data():
    """CSV 파일의 수기 백테스팅 데이터 분석"""
    
    # CSV 파일 로드
    df = pd.read_csv('SAMPLE.csv', encoding='utf-8-sig')
    
    print("=== CSV 데이터 분석 ===")
    print(f"컬럼: {list(df.columns)}")
    print(f"데이터 수: {len(df)}개")
    print(f"기간: {df.iloc[-1]['일자']} ~ {df.iloc[0]['일자']}")
    
    # 계절 분류 확인
    print("\n=== RSI와 계절 매칭 확인 (처음 10개) ===")
    for i in range(min(10, len(df))):
        date = df.iloc[i]['일자']
        rsi = df.iloc[i]['RSI']
        season = df.iloc[i]['사계절']
        
        # 코드 로직으로 계절 계산
        if pd.notna(rsi):
            if rsi >= 70:
                expected = '여름'
            elif rsi >= 50:
                expected = '봄'
            elif rsi >= 30:
                expected = '가을'
            else:
                expected = '겨울'
        else:
            expected = 'N/A'
        
        match = "✓" if expected == season else "✗"
        print(f"{date}: RSI={rsi} → 예상={expected}, 실제={season} {match}")
    
    # 계절별 분포
    season_counts = df['사계절'].value_counts()
    print(f"\n=== 계절별 분포 ===")
    for season, count in season_counts.items():
        print(f"{season}: {count}개")
    
    # 코드 데이터와 매칭 확인
    print("\n=== 코드 S&P500 데이터와 매칭 확인 ===")
    
    # 코드 S&P500 데이터 로드
    sp500_df = pd.read_excel('sp500_data.xlsx')
    sp500_df['날짜'] = pd.to_datetime(sp500_df['날짜'])
    sp500_df = sp500_df.set_index('날짜').sort_index()
    
    print("코드 S&P500 컬럼들:")
    for i, col in enumerate(sp500_df.columns):
        print(f"  {i}: {col}")
    
    # 첫 번째 데이터 비교 (Nov-00)
    sample_date = "Nov-00"
    csv_row = df[df['일자'] == sample_date].iloc[0]
    
    # 2000년 11월 데이터 찾기
    nov_2000 = sp500_df[sp500_df.index.to_period('M') == '2000-11'].iloc[0]
    
    print(f"\n=== {sample_date} 데이터 비교 ===")
    print("CSV 데이터:")
    print(f"  모멘텀: {csv_row['모멘텀']}")
    print(f"  성장: {csv_row['성장']}")
    print(f"  퀄리티: {csv_row['퀄리티']}")
    print(f"  가치: {csv_row['가치']}")
    print(f"  S&P 로볼: {csv_row['S&P 로볼']}")
    
    print("코드 데이터:")
    print(f"  Growth: {nov_2000['S&P500 Growth']}")
    print(f"  Value: {nov_2000['S&P500 Value']}")
    print(f"  Momentum: {nov_2000['S&P500 Momentum']}")
    print(f"  Quality: {nov_2000['S&P500 Quality']}")
    print(f"  Low Vol: {nov_2000['S&P500 Low Volatiltiy Index']}")
    print(f"  Dividend: {nov_2000['S&P500 Div Aristocrt TR Index']}")
    
    # 매칭 확인
    print("\n=== 매칭 결과 ===")
    tolerance = 0.01
    
    matches = []
    if abs(csv_row['모멘텀'] - nov_2000['S&P500 Momentum']) < tolerance:
        matches.append("CSV 모멘텀 = 코드 Momentum")
    if abs(csv_row['성장'] - nov_2000['S&P500 Growth']) < tolerance:
        matches.append("CSV 성장 = 코드 Growth")
    if abs(csv_row['퀄리티'] - nov_2000['S&P500 Quality']) < tolerance:
        matches.append("CSV 퀄리티 = 코드 Quality")
    if abs(csv_row['가치'] - nov_2000['S&P500 Value']) < tolerance:
        matches.append("CSV 가치 = 코드 Value")
    if abs(csv_row['S&P 로볼'] - nov_2000['S&P500 Low Volatiltiy Index']) < tolerance:
        matches.append("CSV S&P 로볼 = 코드 Low Volatility")
    
    for match in matches:
        print(f"  ✓ {match}")

if __name__ == "__main__":
    analyze_csv_data()