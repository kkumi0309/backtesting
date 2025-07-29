import pandas as pd
import numpy as np

# 수기 데이터와 코드 데이터의 정확한 매핑 찾기

def load_manual_sample():
    """수기 샘플 데이터 로드"""
    df = pd.read_excel('SAMPLE.xlsx', skiprows=1)
    
    # 1999-01 데이터 (첫 번째 데이터)
    manual_data = {
        'date': pd.to_datetime(df.iloc[-1, 0]),  # 마지막 행이 1999-01
        'col1_value': df.iloc[-1, 1],     # 1279.64
        'col2_growth': df.iloc[-1, 2],    # 108.84  
        'col3_quality': df.iloc[-1, 3],   # 739.55
        'col4_momentum': df.iloc[-1, 4],  # 102.62
        'col5_unknown': df.iloc[-1, 5],   # 562.02
        'col6_dividend': df.iloc[-1, 6],  # 2443.54
        'rsi': df.iloc[-1, 7]             # 72.54
    }
    
    return manual_data

def load_code_data():
    """코드에서 사용하는 SP500 데이터 로드"""
    df = pd.read_excel('sp500_data.xlsx')
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.set_index('날짜').sort_index()
    
    # 1999-01-01 데이터 찾기
    target_date = pd.Timestamp('1999-01-01')
    
    # 해당 월의 데이터 찾기
    jan_1999 = df[df.index.to_period('M') == '1999-01'].iloc[0]
    
    code_data = {
        'date': jan_1999.name,
        'growth': jan_1999['S&P500 Growth'],        # 739.55
        'value': jan_1999['S&P500 Value'],          # 562.02  
        'momentum': jan_1999['S&P500 Momentum'],    # 108.84
        'quality': jan_1999['S&P500 Quality'],      # 102.62
        'low_vol': jan_1999['S&P500 Low Volatiltiy Index'],  # 2443.54
        'dividend': jan_1999['S&P500 Div Aristocrt TR Index']
    }
    
    return code_data

def find_mapping():
    """수기와 코드 데이터의 정확한 매핑 찾기"""
    manual = load_manual_sample()
    code = load_code_data()
    
    print("=== 1999-01 데이터 비교 ===")
    print(f"날짜: {manual['date']} vs {code['date']}")
    print()
    
    print("수기 데이터:")
    print(f"  컬럼1 (Value?): {manual['col1_value']}")
    print(f"  컬럼2 (Growth?): {manual['col2_growth']}")  
    print(f"  컬럼3 (Quality?): {manual['col3_quality']}")
    print(f"  컬럼4 (Momentum?): {manual['col4_momentum']}")
    print(f"  컬럼5 (Unknown): {manual['col5_unknown']}")
    print(f"  컬럼6 (Dividend?): {manual['col6_dividend']}")
    print()
    
    print("코드 데이터:")
    print(f"  Growth: {code['growth']}")
    print(f"  Value: {code['value']}")
    print(f"  Momentum: {code['momentum']}")
    print(f"  Quality: {code['quality']}")
    print(f"  Low Vol: {code['low_vol']}")
    print(f"  Dividend: {code['dividend']}")
    print()
    
    print("=== 매칭 분석 ===")
    
    # 정확한 매칭 찾기
    mappings = []
    
    if abs(manual['col2_growth'] - code['momentum']) < 0.01:
        mappings.append("수기 컬럼2 (Growth) = 코드 Momentum")
    if abs(manual['col3_quality'] - code['growth']) < 0.01:
        mappings.append("수기 컬럼3 (Quality) = 코드 Growth")
    if abs(manual['col4_momentum'] - code['quality']) < 0.01:
        mappings.append("수기 컬럼4 (Momentum) = 코드 Quality")
    if abs(manual['col5_unknown'] - code['value']) < 0.01:
        mappings.append("수기 컬럼5 (Unknown) = 코드 Value")
    if abs(manual['col6_dividend'] - code['low_vol']) < 0.01:
        mappings.append("수기 컬럼6 (Dividend) = 코드 Low Vol")
    
    print("발견된 매칭:")
    for mapping in mappings:
        print(f"  ✓ {mapping}")
    
    print()
    print("=== 올바른 컬럼 매핑 ===")
    print("수기 백테스팅에서 사용한 컬럼들:")
    print("  컬럼1: 무엇인지 불명 (1279.64)")
    print("  컬럼2: S&P500 Momentum (108.84)")
    print("  컬럼3: S&P500 Growth (739.55)")  
    print("  컬럼4: S&P500 Quality (102.62)")
    print("  컬럼5: S&P500 Value (562.02)")
    print("  컬럼6: S&P500 Low Volatility (2443.54)")

if __name__ == "__main__":
    find_mapping()