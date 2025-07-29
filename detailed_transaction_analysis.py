import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터를 로드합니다."""
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
    
    # 수기 데이터
    manual_df = pd.read_excel('11.xlsx', skiprows=1)
    date_col = manual_df.columns[0]
    manual_df[date_col] = pd.to_datetime(manual_df[date_col])
    manual_df.set_index(date_col, inplace=True)
    manual_df.sort_index(inplace=True)
    
    return sp500_df, rsi_series, manual_df

def analyze_monthly_data_alignment():
    """월별 데이터 정렬 방식을 분석합니다."""
    sp500_df, rsi_series, manual_df = load_data()
    
    print("=== 월별 데이터 정렬 분석 ===")
    
    # 1999년 1월부터 12월까지 상세 분석
    start = pd.Timestamp(1999, 1, 1)
    end = pd.Timestamp(1999, 12, 31)
    
    sp500_1999 = sp500_df.loc[start:end]
    rsi_1999 = rsi_series.loc[start:end]
    manual_1999 = manual_df.loc[start:end] if len(manual_df) > 0 else None
    
    print(f"\n1999년 데이터 포인트:")
    print(f"S&P 500: {len(sp500_1999)}개")
    print(f"RSI: {len(rsi_1999)}개")
    if manual_1999 is not None:
        print(f"수기: {len(manual_1999)}개")
    
    # 월별 매칭 방식 비교
    print(f"\n월별 날짜 매칭 비교 (1999년):")
    print(f"{'월':<5} {'S&P 500 날짜':<15} {'RSI 날짜':<15} {'RSI 값':<8} {'계절':<8}")
    print("-" * 60)
    
    for month in range(1, 13):
        # S&P 500에서 해당 월 데이터
        month_start = pd.Timestamp(1999, month, 1)
        month_end = pd.Timestamp(1999, month, 28) + pd.DateOffset(days=10)  # 월말 처리
        
        sp500_month = sp500_1999.loc[month_start:month_end]
        rsi_month = rsi_1999.loc[month_start:month_end]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            sp500_date = sp500_month.index[0]  # 첫 번째 날짜
            rsi_date = rsi_month.index[0]  # 첫 번째 날짜
            rsi_value = rsi_month.iloc[0]
            
            # 계절 분류
            if rsi_value >= 70:
                season = '여름'
            elif rsi_value >= 50:
                season = '봄'
            elif rsi_value >= 30:
                season = '가을'
            else:
                season = '겨울'
            
            print(f"{month:2d}월  {sp500_date.strftime('%Y-%m-%d'):<15} {rsi_date.strftime('%Y-%m-%d'):<15} {rsi_value:6.2f}  {season:<8}")

def analyze_price_application_methods():
    """가격 적용 방식을 분석합니다."""
    sp500_df, rsi_series, manual_df = load_data()
    
    print(f"\n=== 가격 적용 방식 분석 ===")
    
    # 전략 규칙
    strategy_rules = {
        '봄': 'S&P500 Quality',
        '여름': 'S&P500 Momentum', 
        '가을': 'S&P500 Low Volatiltiy Index',
        '겨울': 'S&P500 Low Volatiltiy Index'
    }
    
    # 1999년 첫 6개월 상세 분석
    start_date = pd.Timestamp(1999, 1, 1)
    end_date = pd.Timestamp(1999, 6, 30)
    
    # 데이터 정렬
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    # 월별 매칭
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    common_periods = sorted(list(set(sp500_filtered['year_month']).intersection(set(rsi_filtered_df['year_month']))))[:6]
    
    print(f"월별 거래 시뮬레이션 (방법 비교):")
    print(f"{'월':<8} {'RSI':<6} {'계절':<6} {'전략':<15} {'방법1가격':<10} {'방법2가격':<10} {'차이%':<8}")
    print("-" * 80)
    
    total_diff = 0
    
    for period in common_periods:
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            rsi_value = rsi_month['RSI'].iloc[0]
            
            # 계절 분류
            if rsi_value >= 70:
                season = '여름'
            elif rsi_value >= 50:
                season = '봄'
            elif rsi_value >= 30:
                season = '가을'
            else:
                season = '겨울'
            
            target_style = strategy_rules[season]
            
            # 방법 1: 첫 번째 날짜 가격 (현재 프로그램 방식)
            price_method1 = sp500_month[target_style].iloc[0]
            
            # 방법 2: 마지막 날짜 가격 (월말 가격)
            price_method2 = sp500_month[target_style].iloc[-1]
            
            # 차이 계산
            price_diff = ((price_method2 - price_method1) / price_method1) * 100
            total_diff += abs(price_diff)
            
            print(f"{str(period):<8} {rsi_value:5.1f} {season:<6} {target_style[:15]:<15} {price_method1:9.2f} {price_method2:9.2f} {price_diff:7.2f}")
    
    print(f"\n평균 가격 차이: {total_diff/len(common_periods):.2f}%")

def test_different_calculation_methods():
    """다양한 계산 방법을 테스트합니다."""
    sp500_df, rsi_series, manual_df = load_data()
    
    print(f"\n=== 계산 방법 비교 테스트 ===")
    
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
    
    # 데이터 필터링
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    # 월별 매칭
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    common_periods = sorted(list(set(sp500_filtered['year_month']).intersection(set(rsi_filtered_df['year_month']))))
    
    results = {}
    
    # 방법 1: 현재 프로그램 방식 (월초 가격)
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    for i, period in enumerate(common_periods):
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            rsi_value = rsi_month['RSI'].iloc[0]
            season = classify_season(rsi_value)
            
            if pd.notna(season):
                target_style = strategy_rules[season]
                price = sp500_month[target_style].iloc[0]  # 월초 가격
                
                if i == 0:
                    current_style = target_style
                    current_shares = portfolio_value / price
                    portfolio_value = current_shares * price
                elif target_style != current_style:
                    # 매도 후 매수
                    if current_style and current_shares > 0:
                        sell_price = sp500_month[current_style].iloc[0]
                        cash = current_shares * sell_price
                        
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                else:
                    portfolio_value = current_shares * price
    
    method1_return = (portfolio_value / initial_capital - 1) * 100
    results['방법1 (월초가격)'] = method1_return
    
    # 방법 2: 월말 가격 적용
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    for i, period in enumerate(common_periods):
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            rsi_value = rsi_month['RSI'].iloc[0]
            season = classify_season(rsi_value)
            
            if pd.notna(season):
                target_style = strategy_rules[season]
                price = sp500_month[target_style].iloc[-1]  # 월말 가격
                
                if i == 0:
                    current_style = target_style
                    current_shares = portfolio_value / price
                    portfolio_value = current_shares * price
                elif target_style != current_style:
                    if current_style and current_shares > 0:
                        sell_price = sp500_month[current_style].iloc[-1]
                        cash = current_shares * sell_price
                        
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                else:
                    portfolio_value = current_shares * price
    
    method2_return = (portfolio_value / initial_capital - 1) * 100
    results['방법2 (월말가격)'] = method2_return
    
    # 방법 3: 평균 가격 적용
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    for i, period in enumerate(common_periods):
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            rsi_value = rsi_month['RSI'].iloc[0]
            season = classify_season(rsi_value)
            
            if pd.notna(season):
                target_style = strategy_rules[season]
                price = sp500_month[target_style].mean()  # 평균 가격
                
                if i == 0:
                    current_style = target_style
                    current_shares = portfolio_value / price
                    portfolio_value = current_shares * price
                elif target_style != current_style:
                    if current_style and current_shares > 0:
                        sell_price = sp500_month[current_style].mean()
                        cash = current_shares * sell_price
                        
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                else:
                    portfolio_value = current_shares * price
    
    method3_return = (portfolio_value / initial_capital - 1) * 100
    results['방법3 (평균가격)'] = method3_return
    
    print(f"계산 방법별 결과 비교:")
    print(f"{'방법':<15} {'총 수익률':<10} {'수기와 차이':<12}")
    print("-" * 40)
    
    manual_return = 1139.95
    for method, return_val in results.items():
        diff = abs(manual_return - return_val)
        print(f"{method:<15} {return_val:8.2f}% {diff:10.2f}%p")
    
    print(f"\n수기 작업 결과: {manual_return:.2f}%")
    
    # 가장 가까운 방법 찾기
    closest_method = min(results.items(), key=lambda x: abs(manual_return - x[1]))
    print(f"가장 가까운 방법: {closest_method[0]} (차이: {abs(manual_return - closest_method[1]):.2f}%p)")
    
    return results

if __name__ == "__main__":
    analyze_monthly_data_alignment()
    analyze_price_application_methods()
    test_different_calculation_methods()