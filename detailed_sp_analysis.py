import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_sp_simulation_detailed():
    """S&P 시뮬레이션 파일을 상세 분석합니다."""
    
    print("=== S&P 시뮬레이션.xlsx 상세 분석 ===")
    
    # 여러 방법으로 파일 읽기 시도
    try:
        # 방법 1: 기본 읽기
        df1 = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name='작업용')
        print("방법 1 - 기본 읽기:")
        print(f"Shape: {df1.shape}")
        print(f"컬럼들: {list(df1.columns)}")
        print("\n첫 5행:")
        print(df1.head())
        
        # 방법 2: 헤더 지정하여 읽기
        print("\n" + "="*60)
        df2 = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name='작업용', header=1)
        print("방법 2 - 두 번째 행을 헤더로:")
        print(f"Shape: {df2.shape}")
        print(f"컬럼들: {list(df2.columns)}")
        print("\n첫 5행:")
        print(df2.head())
        
        # 방법 3: 헤더 없이 읽기
        print("\n" + "="*60)
        df3 = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name='작업용', header=None)
        print("방법 3 - 헤더 없이 읽기:")
        print(f"Shape: {df3.shape}")
        print(f"컬럼들: {list(df3.columns)}")
        print("\n첫 10행:")
        print(df3.head(10))
        
        # 가장 적절한 방법 선택
        return analyze_best_format(df1, df2, df3)
        
    except Exception as e:
        print(f"파일 분석 오류: {e}")
        return None

def analyze_best_format(df1, df2, df3):
    """가장 적절한 데이터 형식을 선택하고 분석합니다."""
    
    print("\n=== 최적 형식 선택 및 분석 ===")
    
    # 각 방법별로 유효한 데이터 확인
    candidates = [
        ("방법1_기본", df1),
        ("방법2_헤더1", df2), 
        ("방법3_헤더없음", df3)
    ]
    
    best_df = None
    best_name = None
    
    for name, df in candidates:
        print(f"\n{name} 분석:")
        
        # 날짜 컬럼 찾기
        date_cols = []
        for i, col in enumerate(df.columns):
            try:
                # 첫 번째 비어있지 않은 값 확인
                first_valid = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if first_valid and (isinstance(first_valid, pd.Timestamp) or 
                                  (isinstance(first_valid, str) and ('2025' in str(first_valid) or '1999' in str(first_valid)))):
                    date_cols.append((i, col))
            except:
                pass
        
        print(f"  날짜 후보 컬럼: {date_cols}")
        
        # 숫자 데이터가 많은 컬럼 확인
        numeric_cols = []
        for col in df.columns:
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count > len(df) * 0.5:  # 50% 이상이 숫자
                numeric_cols.append(col)
        
        print(f"  숫자 데이터 컬럼 수: {len(numeric_cols)}")
        
        # 적절한 데이터 구조인지 판단
        if len(date_cols) > 0 and len(numeric_cols) > 5:
            best_df = df
            best_name = name
            print(f"  ✓ {name}을 최적 형식으로 선택")
            break
    
    if best_df is None:
        print("적절한 데이터 형식을 찾을 수 없습니다.")
        return None
    
    # 선택된 형식으로 상세 분석
    return analyze_selected_format(best_df, best_name)

def analyze_selected_format(df, format_name):
    """선택된 형식의 데이터를 상세 분석합니다."""
    
    print(f"\n=== {format_name} 상세 분석 ===")
    
    # 날짜 컬럼 설정
    date_col = None
    for col in df.columns:
        try:
            if df[col].dtype == 'datetime64[ns]' or any('2025' in str(val) or '1999' in str(val) 
                                                       for val in df[col].dropna()[:5]):
                date_col = col
                break
        except:
            continue
    
    if date_col:
        print(f"날짜 컬럼: {date_col}")
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df_clean = df.set_index(date_col).sort_index()
            print(f"날짜 범위: {df_clean.index.min()} ~ {df_clean.index.max()}")
            print(f"데이터 포인트: {len(df_clean)}개")
        except Exception as e:
            print(f"날짜 처리 오류: {e}")
            df_clean = df
    else:
        print("날짜 컬럼을 찾을 수 없습니다.")
        df_clean = df
    
    # 숫자 컬럼들 분석
    numeric_cols = []
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64'] or pd.to_numeric(df_clean[col], errors='coerce').notna().sum() > len(df_clean) * 0.8:
            numeric_cols.append(col)
    
    print(f"\n숫자 데이터 컬럼들 ({len(numeric_cols)}개):")
    for i, col in enumerate(numeric_cols[:10]):  # 처음 10개만 표시
        try:
            numeric_data = pd.to_numeric(df_clean[col], errors='coerce').dropna()
            if len(numeric_data) > 0:
                print(f"  {i+1:2d}. {col}: 평균={numeric_data.mean():.2f}, 범위={numeric_data.min():.2f}~{numeric_data.max():.2f}")
        except:
            pass
    
    # 포트폴리오 수익률이나 가치로 보이는 컬럼 찾기
    portfolio_candidates = []
    
    for col in df_clean.columns:
        col_str = str(col).lower()
        if any(keyword in col_str for keyword in ['수익률', '가치', '포트폴리오', '총액', '누적', 'return', 'value', 'portfolio']):
            try:
                data = pd.to_numeric(df_clean[col], errors='coerce').dropna()
                if len(data) > 50:  # 충분한 데이터가 있는 경우
                    portfolio_candidates.append((col, data))
            except:
                pass
    
    print(f"\n포트폴리오 후보 컬럼들:")
    for col, data in portfolio_candidates:
        initial = data.iloc[0] if len(data) > 0 else 0
        final = data.iloc[-1] if len(data) > 0 else 0
        if initial != 0:
            return_pct = ((final / initial) - 1) * 100
            print(f"  {col}: {initial:.2f} → {final:.2f} (수익률: {return_pct:.2f}%)")
    
    # 1139.95%와 가장 가까운 결과 찾기
    target = 1139.95
    closest_col = None
    closest_return = None
    min_diff = float('inf')
    
    for col, data in portfolio_candidates:
        if len(data) > 0:
            initial = data.iloc[0]
            final = data.iloc[-1]
            if initial != 0 and initial > 0:
                return_pct = ((final / initial) - 1) * 100
                diff = abs(return_pct - target)
                if diff < min_diff:
                    min_diff = diff
                    closest_col = col
                    closest_return = return_pct
    
    print(f"\n=== 수기 계산(1139.95%)과 비교 ===")
    if closest_col:
        print(f"가장 가까운 컬럼: {closest_col}")
        print(f"해당 수익률: {closest_return:.2f}%")
        print(f"차이: {min_diff:.2f}%p")
        
        if min_diff < 50:  # 50%p 이내 차이
            print("✓ 수기 계산과 매우 유사한 결과 발견!")
        else:
            print("✗ 여전히 큰 차이 존재")
    else:
        print("적절한 포트폴리오 수익률 컬럼을 찾을 수 없습니다.")
    
    # 전체 컬럼 정보 출력 (디버깅용)
    print(f"\n=== 전체 컬럼 정보 (디버깅용) ===")
    for i, col in enumerate(df_clean.columns):
        print(f"{i+1:2d}. {col}")
    
    return df_clean, closest_col, closest_return

def extract_manual_calculation_logic():
    """수기 계산 로직을 추출해봅니다."""
    
    print("\n=== 수기 계산 로직 추출 시도 ===")
    
    # 시뮬레이션 파일에서 RSI나 계절 정보 찾기
    try:
        # 모든 시트 확인
        xl_file = pd.ExcelFile('S&P 시뮬레이션.xlsx')
        
        for sheet_name in xl_file.sheet_names:
            print(f"\n시트 '{sheet_name}' 분석:")
            
            # 여러 방법으로 읽기
            for header_row in [None, 0, 1, 2]:
                try:
                    df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name, header=header_row)
                    
                    # RSI 관련 데이터 찾기
                    rsi_found = False
                    for col in df.columns:
                        if 'RSI' in str(col).upper():
                            rsi_data = pd.to_numeric(df[col], errors='coerce').dropna()
                            if len(rsi_data) > 10:  # 충분한 RSI 데이터가 있는 경우
                                print(f"  RSI 데이터 발견 (header={header_row}): {col}")
                                print(f"    범위: {rsi_data.min():.2f} ~ {rsi_data.max():.2f}")
                                print(f"    데이터 수: {len(rsi_data)}개")
                                rsi_found = True
                                break
                    
                    # 계절 관련 데이터 찾기
                    season_found = False
                    for col in df.columns:
                        col_str = str(col)
                        if any(season in col_str for season in ['봄', '여름', '가을', '겨울']):
                            print(f"  계절 데이터 발견 (header={header_row}): {col}")
                            season_found = True
                    
                    if rsi_found or season_found:
                        print(f"  ✓ 유용한 데이터 발견 (header={header_row})")
                        
                except Exception as e:
                    continue
                    
    except Exception as e:
        print(f"시트 분석 오류: {e}")

if __name__ == "__main__":
    # 상세 분석 실행
    result = analyze_sp_simulation_detailed()
    
    # 수기 계산 로직 추출 시도
    extract_manual_calculation_logic()