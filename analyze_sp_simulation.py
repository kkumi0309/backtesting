import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_sp_simulation_file():
    """S&P 시뮬레이션.xlsx 파일을 분석합니다."""
    
    print("=== S&P 시뮬레이션.xlsx 파일 분석 ===")
    
    try:
        # 파일 로딩 - 여러 시트가 있을 수 있으므로 확인
        xl_file = pd.ExcelFile('S&P 시뮬레이션.xlsx')
        print(f"파일의 시트 목록: {xl_file.sheet_names}")
        
        # 각 시트 내용 확인
        for sheet_name in xl_file.sheet_names:
            print(f"\n=== 시트: {sheet_name} ===")
            
            # 첫 몇 행을 읽어서 구조 파악
            df_preview = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name, nrows=5)
            print(f"컬럼: {list(df_preview.columns)}")
            print(f"첫 5행 미리보기:")
            print(df_preview.head())
            
            # 전체 데이터 로딩
            df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name)
            print(f"전체 행 수: {len(df)}")
            
            # 날짜 컬럼 찾기
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or '날짜' in str(col) or 'Date' in str(col):
                    date_cols.append(col)
            
            if date_cols:
                print(f"날짜 관련 컬럼: {date_cols}")
                date_col = date_cols[0]
                
                # 날짜 범위 확인
                if not df[date_col].isna().all():
                    date_range = f"{df[date_col].min()} ~ {df[date_col].max()}"
                    print(f"날짜 범위: {date_range}")
            
            # RSI 관련 컬럼 찾기
            rsi_cols = [col for col in df.columns if 'RSI' in str(col).upper()]
            if rsi_cols:
                print(f"RSI 관련 컬럼: {rsi_cols}")
            
            # 스타일 지수 관련 컬럼 찾기
            style_cols = []
            style_keywords = ['모멘텀', 'Momentum', '퀄리티', 'Quality', '로볼', 'Low Vol', 
                            '가치', 'Value', '성장', 'Growth', '배당', 'Div']
            
            for col in df.columns:
                for keyword in style_keywords:
                    if keyword in str(col):
                        style_cols.append(col)
                        break
            
            if style_cols:
                print(f"스타일 지수 관련 컬럼: {style_cols}")
            
            # 포트폴리오 관련 컬럼 찾기
            portfolio_cols = [col for col in df.columns if any(word in str(col) for word in 
                            ['포트폴리오', '가치', '수익률', '총액', '합계', 'Portfolio', 'Value'])]
            if portfolio_cols:
                print(f"포트폴리오 관련 컬럼: {portfolio_cols}")
            
            print("-" * 60)
        
        return xl_file
        
    except Exception as e:
        print(f"파일 분석 오류: {e}")
        return None

def extract_strategy_data_from_simulation():
    """시뮬레이션 파일에서 전략 데이터를 추출합니다."""
    
    print("\n=== 시뮬레이션 파일에서 전략 데이터 추출 ===")
    
    try:
        xl_file = pd.ExcelFile('S&P 시뮬레이션.xlsx')
        
        # 메인 데이터가 있을 가능성이 높은 시트 선택
        main_sheet = xl_file.sheet_names[0]  # 첫 번째 시트부터 시작
        
        print(f"분석 시트: {main_sheet}")
        
        # 데이터 로딩
        df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=main_sheet)
        
        # 날짜 컬럼 설정
        date_col = df.columns[0]  # 보통 첫 번째 컬럼이 날짜
        
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            print(f"날짜 설정 완료: {df.index.min()} ~ {df.index.max()}")
        except:
            print("날짜 컬럼 처리 실패, 인덱스 설정 없이 진행")
        
        # RSI 기반 계절 분류가 있는지 확인
        season_cols = [col for col in df.columns if any(season in str(col) for season in 
                      ['봄', '여름', '가을', '겨울', 'Spring', 'Summer', 'Fall', 'Winter'])]
        
        if season_cols:
            print(f"계절 분류 컬럼 발견: {season_cols}")
        
        # 포트폴리오 가치나 수익률 컬럼 찾기
        portfolio_value_cols = []
        for col in df.columns:
            col_str = str(col).lower()
            if any(word in col_str for word in ['포트폴리오', 'portfolio', '가치', 'value', 
                                               '총액', '합계', '누적']):
                portfolio_value_cols.append(col)
        
        if portfolio_value_cols:
            print(f"포트폴리오 가치/수익률 컬럼: {portfolio_value_cols}")
            
            # 첫 번째 포트폴리오 컬럼으로 수익률 계산
            portfolio_col = portfolio_value_cols[0]
            portfolio_data = df[portfolio_col].dropna()
            
            if len(portfolio_data) > 0:
                initial_value = portfolio_data.iloc[0]
                final_value = portfolio_data.iloc[-1]
                
                if initial_value != 0:
                    total_return = ((final_value / initial_value) - 1) * 100
                    print(f"\n시뮬레이션 파일 기준 수익률:")
                    print(f"초기값: {initial_value:,.2f}")
                    print(f"최종값: {final_value:,.2f}")
                    print(f"총 수익률: {total_return:.2f}%")
                    
                    return df, total_return
        
        return df, None
        
    except Exception as e:
        print(f"데이터 추출 오류: {e}")
        return None, None

def compare_with_simulation_file():
    """시뮬레이션 파일 결과와 프로그램 결과를 비교합니다."""
    
    print("\n=== 시뮬레이션 파일 vs 프로그램 결과 비교 ===")
    
    # 시뮬레이션 파일 데이터 추출
    sim_df, sim_return = extract_strategy_data_from_simulation()
    
    if sim_df is None:
        print("시뮬레이션 파일 분석 실패")
        return
    
    # 기존 프로그램 결과
    program_return = 1027.58
    manual_return = 1139.95
    
    print(f"\n결과 비교:")
    print(f"{'구분':<15} {'수익률':<12} {'차이(vs수기)':<15}")
    print("-" * 45)
    print(f"{'수기 계산':<15} {manual_return:>10.2f}% {0:>13.2f}%p")
    print(f"{'기존 프로그램':<15} {program_return:>10.2f}% {manual_return-program_return:>13.2f}%p")
    
    if sim_return is not None:
        diff_vs_manual = manual_return - sim_return
        print(f"{'시뮬레이션파일':<15} {sim_return:>10.2f}% {diff_vs_manual:>13.2f}%p")
        
        # 어느 것이 수기 계산에 더 가까운지 확인
        program_diff = abs(manual_return - program_return)
        sim_diff = abs(manual_return - sim_return)
        
        print(f"\n수기 계산과의 절대 차이:")
        print(f"기존 프로그램: {program_diff:.2f}%p")
        print(f"시뮬레이션 파일: {sim_diff:.2f}%p")
        
        if sim_diff < program_diff:
            print("✓ 시뮬레이션 파일이 수기 계산에 더 가까움")
        else:
            print("✓ 기존 프로그램이 수기 계산에 더 가까움")
    
    return sim_df, sim_return

def replicate_simulation_logic():
    """시뮬레이션 파일의 로직을 재현해봅니다."""
    
    print("\n=== 시뮬레이션 로직 재현 시도 ===")
    
    # 시뮬레이션 파일 분석
    sim_df, sim_return = extract_strategy_data_from_simulation()
    
    if sim_df is None:
        return
    
    # 시뮬레이션 파일에서 사용된 것으로 보이는 데이터 구조 분석
    print("시뮬레이션 파일 데이터 구조 분석:")
    print(f"컬럼 수: {len(sim_df.columns)}")
    print(f"행 수: {len(sim_df)}")
    
    # RSI와 스타일 지수 데이터가 모두 있는지 확인
    has_rsi = any('RSI' in str(col).upper() for col in sim_df.columns)
    has_styles = any(style in str(col) for col in sim_df.columns 
                    for style in ['모멘텀', 'Momentum', '퀄리티', 'Quality'])
    
    print(f"RSI 데이터 포함: {has_rsi}")
    print(f"스타일 지수 데이터 포함: {has_styles}")
    
    if has_rsi and has_styles:
        print("✓ 완전한 백테스팅 데이터 구조 확인")
        
        # 시뮬레이션 파일 기반으로 백테스팅 재실행
        try:
            result = run_backtest_with_simulation_data(sim_df)
            if result:
                print(f"\n재현된 백테스팅 결과: {result:.2f}%")
                print(f"원본 시뮬레이션과 차이: {abs(result - sim_return):.2f}%p")
        except Exception as e:
            print(f"백테스팅 재현 실패: {e}")
    
    return sim_df

def run_backtest_with_simulation_data(sim_df):
    """시뮬레이션 데이터로 백테스팅을 실행합니다."""
    
    # RSI 컬럼 찾기
    rsi_col = None
    for col in sim_df.columns:
        if 'RSI' in str(col).upper():
            rsi_col = col
            break
    
    if rsi_col is None:
        print("RSI 컬럼을 찾을 수 없습니다.")
        return None
    
    # 스타일 지수 컬럼들 매핑
    style_mapping = {}
    
    # 퀄리티
    for col in sim_df.columns:
        if '퀄리티' in str(col) or 'Quality' in str(col):
            style_mapping['봄'] = col
            break
    
    # 모멘텀
    for col in sim_df.columns:
        if '모멘텀' in str(col) or 'Momentum' in str(col):
            style_mapping['여름'] = col
            break
    
    # 로우볼
    for col in sim_df.columns:
        if '로볼' in str(col) or 'Low Vol' in str(col) or 'Volatiltiy' in str(col):
            style_mapping['가을'] = col
            style_mapping['겨울'] = col
            break
    
    if len(style_mapping) < 3:  # 최소 3개 스타일은 있어야 함
        print(f"스타일 매핑 불완전: {style_mapping}")
        return None
    
    print(f"스타일 매핑: {style_mapping}")
    
    # 백테스팅 실행
    initial_capital = 10000000
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    rsi_data = sim_df[rsi_col].dropna()
    
    for i, (date, rsi_value) in enumerate(rsi_data.items()):
        # 계절 분류
        if rsi_value >= 70:
            season = '여름'
        elif rsi_value >= 50:
            season = '봄'
        elif rsi_value >= 30:
            season = '가을'
        else:
            season = '겨울'
        
        if season not in style_mapping:
            continue
            
        target_style = style_mapping[season]
        
        if target_style not in sim_df.columns or date not in sim_df.index:
            continue
            
        price = sim_df.loc[date, target_style]
        
        if pd.isna(price) or price <= 0:
            continue
        
        # 거래 로직
        if i == 0:
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
        elif target_style != current_style:
            # 매도 후 매수
            if current_style and current_shares > 0 and current_style in sim_df.columns:
                sell_price = sim_df.loc[date, current_style]
                if pd.notna(sell_price) and sell_price > 0:
                    cash = current_shares * sell_price
                    
                    current_style = target_style
                    current_shares = cash / price
                    portfolio_value = current_shares * price
        else:
            portfolio_value = current_shares * price
    
    total_return = ((portfolio_value / initial_capital) - 1) * 100
    return total_return

if __name__ == "__main__":
    # 파일 분석
    xl_file = analyze_sp_simulation_file()
    
    if xl_file:
        # 시뮬레이션 파일과 비교
        sim_df, sim_return = compare_with_simulation_file()
        
        # 로직 재현
        replicate_simulation_logic()