import pandas as pd
import numpy as np

def check_data_mapping():
    """데이터 매핑 확인"""
    
    # 수기 데이터 (1999-01)
    manual_values = {
        'col1': 1279.64,
        'col2': 108.843,      # 수기의 "성장"
        'col3': 739.54999,    # 수기의 "품질" 
        'col4': 102.617,      # 수기의 "모멘텀"
        'col5': 562.02002,    # 수기의 "가치"
        'col6': 2443.54       # 수기의 "배당"
    }
    
    # 코드 데이터 (1999-01)
    code_values = {
        'Growth': 739.54999,
        'Value': 562.02002,
        'Momentum': 108.843,
        'Quality': 102.617,
        'Low_Vol': 2443.54,
        'Dividend': 431.16
    }
    
    print("=== 정확한 매핑 찾기 ===")
    print()
    
    # 매칭 확인
    if abs(manual_values['col2'] - code_values['Momentum']) < 0.01:
        print("수기 컬럼2 (성장) = 코드 Momentum")
    
    if abs(manual_values['col3'] - code_values['Growth']) < 0.01:
        print("수기 컬럼3 (품질) = 코드 Growth")
        
    if abs(manual_values['col4'] - code_values['Quality']) < 0.01:
        print("수기 컬럼4 (모멘텀) = 코드 Quality")
        
    if abs(manual_values['col5'] - code_values['Value']) < 0.01:
        print("수기 컬럼5 (가치) = 코드 Value")
        
    if abs(manual_values['col6'] - code_values['Low_Vol']) < 0.01:
        print("수기 컬럼6 (배당) = 코드 Low Volatility")
    
    print()
    print("=== 수기 백테스팅의 컬럼 라벨과 실제 데이터 불일치 ===")
    print("수기에서 '성장'이라고 표시된 컬럼 → 실제로는 Momentum 데이터")
    print("수기에서 '품질'이라고 표시된 컬럼 → 실제로는 Growth 데이터") 
    print("수기에서 '모멘텀'이라고 표시된 컬럼 → 실제로는 Quality 데이터")
    print("수기에서 '가치'라고 표시된 컬럼 → 실제로는 Value 데이터") 
    print("수기에서 '배당'이라고 표시된 컬럼 → 실제로는 Low Volatility 데이터")
    
    print()
    print("=== 코드 수정 방향 ===")
    print("AA.py의 기본 전략들을 수기 백테스팅에서 실제 사용된 컬럼에 맞춰 수정해야 함")
    print()
    print("예시 - 모멘텀 전략:")
    print("  기존: '여름': 'S&P500 Momentum'")
    print("  수정: '여름': 'S&P500 Momentum' (실제로는 수기 컬럼2)")

if __name__ == "__main__":
    check_data_mapping()