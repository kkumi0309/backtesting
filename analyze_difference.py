import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_manual_data():
    """수기 백테스팅 데이터 로드"""
    df = pd.read_excel('SAMPLE.xlsx', skiprows=1)
    
    # 컬럼 인덱스로 접근 (한글 깨짐 문제)
    data = {
        'date': pd.to_datetime(df.iloc[:, 0]),
        'value': df.iloc[:, 1],      # Value 지수
        'growth': df.iloc[:, 2],     # Growth 지수  
        'quality': df.iloc[:, 3],    # Quality 지수
        'momentum': df.iloc[:, 4],   # Momentum 지수
        'dividend': df.iloc[:, 6],   # Dividend 지수
        'rsi': df.iloc[:, 7],        # RSI
        'season': df.iloc[:, 14]     # 계절
    }
    
    manual_df = pd.DataFrame(data)
    manual_df = manual_df.set_index('date').sort_index()
    
    return manual_df

def load_code_data():
    """코드에서 사용하는 데이터들 로드"""
    # S&P 500 데이터
    sp500_df = pd.read_excel('sp500_data.xlsx')
    date_column = sp500_df.columns[0]
    sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
    sp500_df.set_index(date_column, inplace=True)
    sp500_df.sort_index(inplace=True)
    
    # RSI 데이터
    rsi_df = pd.read_excel('RSI_DATE.xlsx', skiprows=1)
    date_column = rsi_df.columns[0]
    rsi_df[date_column] = pd.to_datetime(rsi_df[date_column])
    rsi_df.set_index(date_column, inplace=True)
    rsi_df.sort_index(inplace=True)
    rsi_series = rsi_df['RSI'].dropna()
    
    return sp500_df, rsi_series

def classify_season(rsi_value):
    """RSI 값으로 계절 분류"""
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

def compare_data():
    """수기와 코드 데이터 비교"""
    print("=== 수기 백테스팅 vs 코드 데이터 비교 ===\n")
    
    # 데이터 로드
    manual_df = load_manual_data()
    sp500_df, rsi_series = load_code_data()
    
    print("1. 데이터 기본 정보:")
    print(f"   수기 데이터 기간: {manual_df.index.min()} ~ {manual_df.index.max()}")
    print(f"   수기 데이터 포인트: {len(manual_df)}개")
    print(f"   코드 S&P500 기간: {sp500_df.index.min()} ~ {sp500_df.index.max()}")
    print(f"   코드 S&P500 포인트: {len(sp500_df)}개")
    print(f"   코드 RSI 기간: {rsi_series.index.min()} ~ {rsi_series.index.max()}")
    print(f"   코드 RSI 포인트: {len(rsi_series)}개")
    
    # 공통 기간 찾기
    common_start = max(manual_df.index.min(), sp500_df.index.min(), rsi_series.index.min())
    common_end = min(manual_df.index.max(), sp500_df.index.max(), rsi_series.index.max())
    
    print(f"\n2. 공통 기간: {common_start} ~ {common_end}")
    
    # 월별 데이터 매칭
    manual_monthly = manual_df.groupby(manual_df.index.to_period('M')).first()
    sp500_monthly = sp500_df.groupby(sp500_df.index.to_period('M')).first()
    rsi_monthly = rsi_series.groupby(rsi_series.index.to_period('M')).first()
    
    # 공통 월 찾기
    common_periods = sorted(list(set(manual_monthly.index) & set(sp500_monthly.index) & set(rsi_monthly.index)))
    
    print(f"   공통 월: {len(common_periods)}개")
    
    if len(common_periods) == 0:
        print("ERROR: 공통 데이터가 없습니다!")
        return
    
    print(f"   첫 공통 월: {common_periods[0]}")
    print(f"   마지막 공통 월: {common_periods[-1]}")
    
    # RSI 값 비교
    print("\n3. RSI 값 비교 (첫 5개월):")
    for i, period in enumerate(common_periods[:5]):
        manual_rsi = manual_monthly.loc[period, 'rsi']
        code_rsi = rsi_monthly.loc[period]
        
        print(f"   {period}: 수기={manual_rsi:.2f}, 코드={code_rsi:.2f}, 차이={abs(manual_rsi-code_rsi):.2f}")
    
    # 계절 분류 비교
    print("\n4. 계절 분류 비교 (첫 5개월):")
    for i, period in enumerate(common_periods[:5]):
        manual_season = str(manual_monthly.loc[period, 'season']).strip()
        manual_rsi = manual_monthly.loc[period, 'rsi']
        code_season = classify_season(manual_rsi)
        
        print(f"   {period}: RSI={manual_rsi:.2f} → 수기계절={manual_season}, 코드계절={code_season}")
    
    # S&P 500 지수 값 비교
    print("\n5. S&P 500 지수 값 비교 (첫 3개월):")
    
    # 컬럼 매핑 확인
    print("   코드 S&P500 컬럼들:")
    for i, col in enumerate(sp500_df.columns):
        print(f"     {i}: {col}")
    
    # 수기 데이터와 매칭되는 컬럼 찾기
    for i, period in enumerate(common_periods[:3]):
        print(f"\n   === {period} ===")
        manual_data = manual_monthly.loc[period]
        code_data = sp500_monthly.loc[period]
        
        print(f"   수기 Value: {manual_data['value']:.2f}")
        print(f"   수기 Growth: {manual_data['growth']:.2f}")
        print(f"   수기 Quality: {manual_data['quality']:.2f}")
        print(f"   수기 Momentum: {manual_data['momentum']:.2f}")
        print(f"   수기 Dividend: {manual_data['dividend']:.2f}")
        
        if len(sp500_df.columns) >= 5:
            print(f"   코드 컬럼0: {code_data.iloc[0]:.2f}")
            print(f"   코드 컬럼1: {code_data.iloc[1]:.2f}")
            print(f"   코드 컬럼2: {code_data.iloc[2]:.2f}")
            print(f"   코드 컬럼3: {code_data.iloc[3]:.2f}")
            print(f"   코드 컬럼4: {code_data.iloc[4]:.2f}")

if __name__ == "__main__":
    compare_data()