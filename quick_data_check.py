import pandas as pd
import numpy as np

def check_data_ranges():
    """데이터 범위를 빠르게 확인합니다."""
    
    # S&P 500 데이터
    print("=== S&P 500 데이터 ===")
    sp500_df = pd.read_excel('sp500_data.xlsx')
    date_column = sp500_df.columns[0]
    sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
    sp500_df.set_index(date_column, inplace=True)
    sp500_df.sort_index(inplace=True)
    
    print(f"범위: {sp500_df.index.min()} ~ {sp500_df.index.max()}")
    print(f"총 {len(sp500_df)}개 포인트")
    print(f"컬럼: {list(sp500_df.columns)}")
    
    # RSI 데이터
    print(f"\n=== RSI 데이터 ===")
    rsi_df = pd.read_excel('RSI_DATE.xlsx', skiprows=1)
    date_column = rsi_df.columns[0]
    rsi_df[date_column] = pd.to_datetime(rsi_df[date_column])
    rsi_df.set_index(date_column, inplace=True)
    rsi_df.sort_index(inplace=True)
    
    print(f"범위: {rsi_df.index.min()} ~ {rsi_df.index.max()}")
    print(f"총 {len(rsi_df)}개 포인트")
    print(f"컬럼: {list(rsi_df.columns)}")
    
    # 수기 데이터
    print(f"\n=== 수기 데이터 ===")
    manual_df = pd.read_excel('11.xlsx', skiprows=1)
    date_col = manual_df.columns[0]
    manual_df[date_col] = pd.to_datetime(manual_df[date_col])
    manual_df.set_index(date_col, inplace=True)
    manual_df.sort_index(inplace=True)
    
    print(f"범위: {manual_df.index.min()} ~ {manual_df.index.max()}")
    print(f"총 {len(manual_df)}개 포인트")
    
    # 공통 범위 찾기
    all_start = max(sp500_df.index.min(), rsi_df.index.min(), manual_df.index.min())
    all_end = min(sp500_df.index.max(), rsi_df.index.max(), manual_df.index.max())
    
    print(f"\n=== 공통 범위 ===")
    print(f"시작: {all_start}")
    print(f"종료: {all_end}")
    
    # 실제 1999년 데이터 존재 여부 확인
    year_1999_start = pd.Timestamp(1999, 1, 1)
    year_1999_end = pd.Timestamp(1999, 12, 31)
    
    sp500_1999_exists = any((sp500_df.index >= year_1999_start) & (sp500_df.index <= year_1999_end))
    rsi_1999_exists = any((rsi_df.index >= year_1999_start) & (rsi_df.index <= year_1999_end))
    
    print(f"\n=== 1999년 데이터 존재 여부 ===")
    print(f"S&P 500: {sp500_1999_exists}")
    print(f"RSI: {rsi_1999_exists}")
    
    if sp500_1999_exists:
        sp500_1999 = sp500_df[(sp500_df.index >= year_1999_start) & (sp500_df.index <= year_1999_end)]
        print(f"S&P 500 1999년: {len(sp500_1999)}개 포인트")
        print(f"첫 날짜: {sp500_1999.index.min()}")
        print(f"마지막 날짜: {sp500_1999.index.max()}")
    
    if rsi_1999_exists:
        rsi_1999 = rsi_df[(rsi_df.index >= year_1999_start) & (rsi_df.index <= year_1999_end)]
        print(f"RSI 1999년: {len(rsi_1999)}개 포인트")
        print(f"첫 날짜: {rsi_1999.index.min()}")  
        print(f"마지막 날짜: {rsi_1999.index.max()}")

if __name__ == "__main__":
    check_data_ranges()