import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def find_simulation_calculation_method():
    """시뮬레이션 파일에서 1139.95%를 만드는 정확한 방법을 찾습니다."""
    
    print("=== 시뮬레이션 파일 1139.95% 계산 방법 역추적 ===")
    
    # 기존 프로그램과 시뮬레이션 파일 데이터 로딩
    try:
        # 기존 데이터 로딩
        sp500_df = pd.read_excel('sp500_data.xlsx')
        date_column = sp500_df.columns[0]
        sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
        sp500_df.set_index(date_column, inplace=True)
        sp500_df.sort_index(inplace=True)
        
        rsi_df = pd.read_excel('RSI_DATE.xlsx', skiprows=1)
        date_column = rsi_df.columns[0]
        rsi_df[date_column] = pd.to_datetime(rsi_df[date_column])
        rsi_df.set_index(date_column, inplace=True)
        rsi_df.sort_index(inplace=True)
        rsi_series = rsi_df['RSI'].dropna()
        
        # 시뮬레이션 파일 (header=1이 RSI 데이터 있는 형태)
        sim_df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=0, header=1)
        sim_date_col = sim_df.columns[0]
        sim_df[sim_date_col] = pd.to_datetime(sim_df[sim_date_col])
        sim_df.set_index(sim_date_col, inplace=True)
        sim_df.sort_index(inplace=True)
        
        print("데이터 로딩 완료")
        print(f"기존 S&P500: {len(sp500_df)}개 행")
        print(f"기존 RSI: {len(rsi_series)}개 행")
        print(f"시뮬레이션: {len(sim_df)}개 행")
        
        # 데이터 비교 분석
        return compare_calculation_methods(sp500_df, rsi_series, sim_df)
        
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return None

def compare_calculation_methods(sp500_df, rsi_series, sim_df):
    """다양한 계산 방법을 비교합니다."""
    
    print(f"\n=== 계산 방법 비교 분석 ===")
    
    # 기간 설정
    start_date = pd.Timestamp(1999, 1, 1)
    end_date = pd.Timestamp(2025, 6, 30)
    initial_capital = 10000000
    
    # 전략 규칙
    strategy_rules = {
        '봄': 'S&P500 Quality',
        '여름': 'S&P500 Momentum', 
        '가을': 'S&P500 Low Volatiltiy Index',
        '겨울': 'S&P500 Low Volatiltiy Index'
    }
    
    # 시뮬레이션 파일 매핑 (컬럼명이 다를 수 있음)
    sim_strategy_rules = {
        '봄': '퀄리티',
        '여름': '모멘텀',
        '가을': 'S&P 로볼',
        '겨울': 'S&P 로볼'
    }
    
    def classify_season(rsi_value):
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
    
    # 방법 1: 기존 프로그램 방식
    print("방법 1: 기존 프로그램 방식")
    sp500_filtered = sp500_df.loc[start_date:end_date]
    rsi_filtered = rsi_series.loc[start_date:end_date]
    
    result1 = run_standard_backtest(sp500_filtered, rsi_filtered, strategy_rules, initial_capital, classify_season)
    print(f"결과 1: {result1:.2f}%")
    
    # 방법 2: 시뮬레이션 파일 데이터 직접 사용
    print("\n방법 2: 시뮬레이션 파일 데이터 사용")
    sim_filtered = sim_df.loc[start_date:end_date]
    
    if 'RSI' in sim_filtered.columns:
        sim_rsi = pd.to_numeric(sim_filtered['RSI'], errors='coerce').dropna()
        result2 = run_simulation_backtest(sim_filtered, sim_rsi, sim_strategy_rules, initial_capital, classify_season)
        print(f"결과 2: {result2:.2f}%")
    else:
        result2 = None
        print("결과 2: RSI 컬럼 없음")
    
    # 방법 3: 데이터 정밀도 최대화
    print("\n방법 3: 고정밀도 계산")
    result3 = run_high_precision_backtest(sp500_filtered, rsi_filtered, strategy_rules, initial_capital, classify_season)
    print(f"결과 3: {result3:.2f}%")
    
    # 방법 4: 복합 방법 (시뮬레이션 RSI + 기존 가격)
    print("\n방법 4: 복합 방법")
    if 'RSI' in sim_filtered.columns:
        sim_rsi_aligned = align_rsi_data(sim_filtered, sp500_filtered)
        result4 = run_hybrid_backtest(sp500_filtered, sim_rsi_aligned, strategy_rules, initial_capital, classify_season)
        print(f"결과 4: {result4:.2f}%")
    else:
        result4 = None
        print("결과 4: 실행 불가")
    
    # 결과 비교
    results = {
        '기존 프로그램': result1,
        '시뮬레이션 직접': result2,
        '고정밀도': result3,
        '복합 방법': result4
    }
    
    target = 1139.95
    print(f"\n=== 수기 계산 {target}%와 비교 ===")
    print(f"{'방법':<15} {'수익률':<12} {'차이':<10} {'상태'}")
    print("-" * 45)
    
    best_method = None
    min_diff = float('inf')
    
    for method_name, result in results.items():
        if result is not None:
            diff = abs(result - target)
            status = "★★★" if diff < 5 else "★★" if diff < 20 else "★" if diff < 50 else ""
            print(f"{method_name:<15} {result:>10.2f}% {diff:>8.2f}%p {status}")
            
            if diff < min_diff:
                min_diff = diff
                best_method = method_name
        else:
            print(f"{method_name:<15} {'N/A':>10} {'N/A':>8}")
    
    if best_method and min_diff < 20:
        print(f"\n★ 최적 방법: {best_method} (차이: {min_diff:.2f}%p)")
        return best_method, results[best_method]
    else:
        print(f"\n적절한 방법을 찾지 못했습니다. 추가 분석이 필요합니다.")
        return None, None

def run_standard_backtest(sp500_df, rsi_series, strategy_rules, initial_capital, classify_season):
    """기존 표준 백테스팅"""
    
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    seasons = rsi_series.apply(classify_season)
    
    for i, date in enumerate(sp500_df.index):
        if date not in seasons.index:
            continue
            
        season = seasons.loc[date]
        if pd.isna(season):
            continue
            
        target_style = strategy_rules[season]
        price = sp500_df.loc[date, target_style]
        
        if i == 0:
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
        elif target_style != current_style:
            if current_style and current_shares > 0:
                sell_price = sp500_df.loc[date, current_style]
                cash = current_shares * sell_price
                current_style = target_style
                current_shares = cash / price
                portfolio_value = current_shares * price
        else:
            portfolio_value = current_shares * price
    
    return (portfolio_value / initial_capital - 1) * 100

def run_simulation_backtest(sim_df, sim_rsi, strategy_rules, initial_capital, classify_season):
    """시뮬레이션 파일 데이터로 백테스팅"""
    
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    for i, (date, rsi_value) in enumerate(sim_rsi.items()):
        season = classify_season(rsi_value)
        if pd.isna(season):
            continue
            
        target_style = strategy_rules[season]
        
        if target_style not in sim_df.columns or date not in sim_df.index:
            continue
            
        try:
            price = pd.to_numeric(sim_df.loc[date, target_style], errors='coerce')
            if pd.isna(price) or price <= 0:
                continue
        except:
            continue
        
        if i == 0:
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
        elif target_style != current_style:
            if current_style and current_shares > 0 and current_style in sim_df.columns:
                try:
                    sell_price = pd.to_numeric(sim_df.loc[date, current_style], errors='coerce')
                    if pd.notna(sell_price) and sell_price > 0:
                        cash = current_shares * sell_price
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                except:
                    pass
        else:
            portfolio_value = current_shares * price
    
    return (portfolio_value / initial_capital - 1) * 100

def run_high_precision_backtest(sp500_df, rsi_series, strategy_rules, initial_capital, classify_season):
    """고정밀도 백테스팅"""
    
    from decimal import Decimal, getcontext
    getcontext().prec = 50  # 50자리 정밀도
    
    portfolio_value = Decimal(str(initial_capital))
    current_shares = Decimal('0')
    current_style = None
    
    seasons = rsi_series.apply(classify_season)
    
    for i, date in enumerate(sp500_df.index):
        if date not in seasons.index:
            continue
            
        season = seasons.loc[date]
        if pd.isna(season):
            continue
            
        target_style = strategy_rules[season]
        price = Decimal(str(sp500_df.loc[date, target_style]))
        
        if i == 0:
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
        elif target_style != current_style:
            if current_style and current_shares > 0:
                sell_price = Decimal(str(sp500_df.loc[date, current_style]))
                cash = current_shares * sell_price
                current_style = target_style
                current_shares = cash / price
                portfolio_value = current_shares * price
        else:
            portfolio_value = current_shares * price
    
    final_value = float(portfolio_value)
    return (final_value / initial_capital - 1) * 100

def run_hybrid_backtest(sp500_df, aligned_rsi, strategy_rules, initial_capital, classify_season):
    """하이브리드 백테스팅"""
    
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    for i, (date, rsi_value) in enumerate(aligned_rsi.items()):
        season = classify_season(rsi_value)
        if pd.isna(season) or date not in sp500_df.index:
            continue
            
        target_style = strategy_rules[season]
        price = sp500_df.loc[date, target_style]
        
        if i == 0:
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
        elif target_style != current_style:
            if current_style and current_shares > 0:
                sell_price = sp500_df.loc[date, current_style]
                cash = current_shares * sell_price
                current_style = target_style
                current_shares = cash / price
                portfolio_value = current_shares * price
        else:
            portfolio_value = current_shares * price
    
    return (portfolio_value / initial_capital - 1) * 100

def align_rsi_data(sim_df, sp500_df):
    """RSI 데이터 정렬"""
    
    if 'RSI' not in sim_df.columns:
        return pd.Series()
    
    sim_rsi = pd.to_numeric(sim_df['RSI'], errors='coerce').dropna()
    
    # 공통 날짜만 추출
    common_dates = sim_rsi.index.intersection(sp500_df.index)
    return sim_rsi.loc[common_dates]

if __name__ == "__main__":
    best_method, best_result = find_simulation_calculation_method()
    
    if best_method:
        print(f"\n=== 최종 결론 ===")
        print(f"정확한 계산 방법: {best_method}")
        print(f"계산 결과: {best_result:.2f}%")
        print(f"수기 계산과의 차이: {abs(best_result - 1139.95):.2f}%p")
        
        if abs(best_result - 1139.95) < 10:
            print("✓ 수기 계산 방법을 성공적으로 재현했습니다!")
        else:
            print("△ 어느 정도 근사한 결과를 얻었습니다.")
    else:
        print("\n추가 분석이 필요합니다.")