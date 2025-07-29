import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_correct_sheet_name():
    """올바른 시트 이름을 찾습니다."""
    try:
        xl_file = pd.ExcelFile('S&P 시뮬레이션.xlsx')
        sheet_names = xl_file.sheet_names
        print(f"사용 가능한 시트: {sheet_names}")
        return sheet_names[0] if sheet_names else None
    except Exception as e:
        print(f"시트 확인 오류: {e}")
        return None

def extract_simulation_backtest_data():
    """시뮬레이션 파일에서 백테스팅 데이터를 추출합니다."""
    
    print("=== 시뮬레이션 파일 백테스팅 데이터 추출 ===")
    
    sheet_name = get_correct_sheet_name()
    if not sheet_name:
        return None
    
    # 다양한 헤더 옵션으로 시도
    best_df = None
    best_header = None
    
    for header_option in [None, 0, 1, 2]:
        try:
            df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name, header=header_option)
            
            # RSI 컬럼이 있고 유효한 데이터가 있는지 확인
            rsi_col = None
            for col in df.columns:
                if 'RSI' in str(col).upper():
                    rsi_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(rsi_data) > 100:  # 충분한 데이터
                        rsi_col = col
                        break
            
            if rsi_col:
                best_df = df
                best_header = header_option
                print(f"✓ 최적 헤더 옵션: {header_option}")
                print(f"✓ RSI 컬럼: {rsi_col}")
                break
                
        except Exception as e:
            continue
    
    if best_df is None:
        print("RSI 데이터를 포함한 적절한 형식을 찾을 수 없습니다.")
        return None
    
    return analyze_simulation_data(best_df, best_header)

def analyze_simulation_data(df, header_option):
    """시뮬레이션 데이터를 분석합니다."""
    
    print(f"\n=== 시뮬레이션 데이터 상세 분석 (header={header_option}) ===")
    print(f"데이터 형태: {df.shape}")
    print(f"컬럼들: {list(df.columns)}")
    
    # 날짜 컬럼 찾기 및 설정
    date_col = None
    for col in df.columns:
        try:
            # 첫 몇 개 값 확인
            sample_vals = df[col].dropna().head(5)
            for val in sample_vals:
                if isinstance(val, pd.Timestamp) or ('2025' in str(val) or '1999' in str(val)):
                    date_col = col
                    break
            if date_col:
                break
        except:
            continue
    
    if date_col:
        print(f"날짜 컬럼: {date_col}")
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            print(f"날짜 범위: {df.index.min()} ~ {df.index.max()}")
        except Exception as e:
            print(f"날짜 처리 오류: {e}")
    
    # RSI 컬럼 찾기
    rsi_col = None
    for col in df.columns:
        if 'RSI' in str(col).upper():
            rsi_col = col
            break
    
    if not rsi_col:
        print("RSI 컬럼을 찾을 수 없습니다.")
        return None
    
    # RSI 데이터 분석
    rsi_data = pd.to_numeric(df[rsi_col], errors='coerce').dropna()
    print(f"\nRSI 분석:")
    print(f"컬럼명: {rsi_col}")
    print(f"데이터 수: {len(rsi_data)}개")
    print(f"범위: {rsi_data.min():.2f} ~ {rsi_data.max():.2f}")
    print(f"평균: {rsi_data.mean():.2f}")
    
    # 스타일 지수 컬럼들 찾기
    style_columns = {}
    style_keywords = {
        'quality': ['퀄리티', 'Quality', 'QUALITY'],
        'momentum': ['모멘텀', 'Momentum', 'MOMENTUM'],
        'low_vol': ['로볼', 'Low Vol', 'LOW VOL', 'Volatiltiy', 'VOLATILTIY'],
        'value': ['가치', 'Value', 'VALUE'],
        'growth': ['성장', 'Growth', 'GROWTH'],
        'dividend': ['배당', 'Div', 'DIV', 'Aristocrt']
    }
    
    print(f"\n스타일 지수 컬럼 매핑:")
    for style_name, keywords in style_keywords.items():
        for col in df.columns:
            col_str = str(col)
            if any(keyword in col_str for keyword in keywords):
                style_columns[style_name] = col
                print(f"  {style_name}: {col}")
                break
    
    # 수익률이나 포트폴리오 가치 컬럼 찾기
    value_columns = []
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword in col_str for keyword in ['수익률', '가치', '포트폴리오', '총액', '누적', 'return', 'value', 'portfolio']):
            try:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 50:
                    value_columns.append(col)
            except:
                pass
    
    print(f"\n포트폴리오 가치/수익률 후보 컬럼들:")
    target_return = 1139.95
    closest_match = None
    min_diff = float('inf')
    
    for col in value_columns:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0:
                initial = data.iloc[0]
                final = data.iloc[-1]
                
                if initial > 0:
                    # 수익률 계산 방법 1: 단순 비율
                    return1 = ((final / initial) - 1) * 100
                    
                    # 수익률 계산 방법 2: 이미 수익률인 경우
                    return2 = final if final > 100 else final * 100
                    
                    print(f"  {col}:")
                    print(f"    초기값: {initial:.2f}, 최종값: {final:.2f}")
                    print(f"    방법1 (비율): {return1:.2f}%")
                    print(f"    방법2 (직접): {return2:.2f}%")
                    
                    # 1139.95%와 비교
                    diff1 = abs(return1 - target_return)
                    diff2 = abs(return2 - target_return)
                    
                    if diff1 < min_diff:
                        min_diff = diff1
                        closest_match = (col, return1, "비율계산")
                    
                    if diff2 < min_diff:
                        min_diff = diff2
                        closest_match = (col, return2, "직접값")
                        
        except Exception as e:
            print(f"  {col}: 계산 오류 - {e}")
    
    print(f"\n=== 수기 계산 1139.95%와 비교 ===")
    if closest_match:
        col, return_val, method = closest_match
        print(f"가장 가까운 결과: {col} ({method})")
        print(f"계산된 수익률: {return_val:.2f}%")
        print(f"차이: {min_diff:.2f}%p")
        
        if min_diff < 10:
            print("✓ 수기 계산과 거의 일치!")
            return df, rsi_col, style_columns, col, return_val
        elif min_diff < 100:
            print("△ 어느 정도 유사한 결과")
        else:
            print("✗ 여전히 큰 차이")
    
    return df, rsi_col, style_columns, None, None

def replicate_simulation_strategy(df, rsi_col, style_columns):
    """시뮬레이션 파일 기반으로 전략을 재현합니다."""
    
    print(f"\n=== 시뮬레이션 전략 재현 ===")
    
    if not rsi_col or len(style_columns) < 3:
        print("필요한 데이터가 부족합니다.")
        return None
    
    # 전략 매핑 설정
    strategy_mapping = {}
    if 'quality' in style_columns:
        strategy_mapping['봄'] = style_columns['quality']
    if 'momentum' in style_columns:
        strategy_mapping['여름'] = style_columns['momentum']
    if 'low_vol' in style_columns:
        strategy_mapping['가을'] = style_columns['low_vol']
        strategy_mapping['겨울'] = style_columns['low_vol']
    
    if len(strategy_mapping) < 3:
        print(f"전략 매핑 불완전: {strategy_mapping}")
        return None
    
    print(f"전략 매핑: {strategy_mapping}")
    
    # 백테스팅 실행
    initial_capital = 10000000
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    # RSI 데이터 가져오기
    rsi_series = pd.to_numeric(df[rsi_col], errors='coerce').dropna()
    
    print(f"\n거래 시뮬레이션 (처음 10개):")
    print(f"{'날짜':<12} {'RSI':<6} {'계절':<6} {'스타일':<20} {'가격':<10} {'포트폴리오':<15}")
    print("-" * 80)
    
    for i, (date, rsi_value) in enumerate(rsi_series.items()):
        # 계절 분류
        if rsi_value >= 70:
            season = '여름'
        elif rsi_value >= 50:
            season = '봄'
        elif rsi_value >= 30:
            season = '가을'
        else:
            season = '겨울'
        
        if season not in strategy_mapping:
            continue
        
        target_style = strategy_mapping[season]
        
        # 해당 날짜의 가격
        if target_style not in df.columns or date not in df.index:
            continue
        
        try:
            price = pd.to_numeric(df.loc[date, target_style], errors='coerce')
            if pd.isna(price) or price <= 0:
                continue
        except:
            continue
        
        # 거래 로직
        if i == 0:
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
        elif target_style != current_style:
            # 매도 후 매수
            if current_style and current_shares > 0 and current_style in df.columns:
                try:
                    sell_price = pd.to_numeric(df.loc[date, current_style], errors='coerce')
                    if pd.notna(sell_price) and sell_price > 0:
                        cash = current_shares * sell_price
                        
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                except:
                    pass
        else:
            portfolio_value = current_shares * price
        
        # 처음 10개 출력
        if i < 10:
            style_short = target_style[:20] if target_style else "N/A"
            print(f"{date.strftime('%Y-%m') if hasattr(date, 'strftime') else str(date)[:7]:<12} {rsi_value:5.1f} {season:<6} {style_short:<20} {price:9.2f} {portfolio_value:13,.0f}")
    
    # 최종 결과
    total_return = ((portfolio_value / initial_capital) - 1) * 100
    
    print(f"\n재현된 백테스팅 결과:")
    print(f"초기 자본: {initial_capital:,}원")
    print(f"최종 가치: {portfolio_value:,}원")  
    print(f"총 수익률: {total_return:.2f}%")
    
    # 수기 계산과 비교
    manual_return = 1139.95
    diff = abs(total_return - manual_return)
    print(f"\n수기 계산 1139.95%와 비교:")
    print(f"차이: {diff:.2f}%p")
    
    if diff < 50:
        print("✓ 수기 계산과 매우 유사!")
    elif diff < 200:
        print("△ 어느 정도 유사")
    else:
        print("✗ 여전히 큰 차이")
    
    return total_return

if __name__ == "__main__":
    # 시뮬레이션 데이터 추출
    result = extract_simulation_backtest_data()
    
    if result and len(result) == 5:
        df, rsi_col, style_columns, value_col, return_val = result
        
        # 전략 재현
        replicated_return = replicate_simulation_strategy(df, rsi_col, style_columns)
        
        print(f"\n=== 최종 결과 요약 ===")
        print(f"수기 계산: 1139.95%")
        print(f"기존 프로그램: 1027.58%")
        if return_val:
            print(f"시뮬레이션 파일: {return_val:.2f}%")
        if replicated_return:
            print(f"재현된 백테스팅: {replicated_return:.2f}%")