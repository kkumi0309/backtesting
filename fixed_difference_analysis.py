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

def analyze_exact_calculation_difference():
    """정확한 계산 차이를 분석합니다."""
    print("=== 수기 vs 프로그램 계산 차이 정밀 분석 ===")
    
    sp500_df, rsi_series, manual_df = load_data()
    
    # 공통 기간 설정 
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
    
    # 데이터 필터링 및 정렬
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    print(f"필터링된 데이터:")
    print(f"S&P 500: {len(sp500_filtered)}개 ({sp500_filtered.index.min().strftime('%Y-%m')} ~ {sp500_filtered.index.max().strftime('%Y-%m')})")
    print(f"RSI: {len(rsi_filtered)}개 ({rsi_filtered.index.min().strftime('%Y-%m')} ~ {rsi_filtered.index.max().strftime('%Y-%m')})")
    
    # 월별 데이터 매칭
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    # 공통 기간
    sp500_periods = set(sp500_filtered['year_month'])
    rsi_periods = set(rsi_filtered_df['year_month'])
    common_periods = sorted(list(sp500_periods.intersection(rsi_periods)))
    
    print(f"공통 월별 기간: {len(common_periods)}개")
    
    # 수기 작업에서 사용한 것으로 추정되는 데이터 확인
    print(f"\n=== 수기 데이터 분석 ===")
    if 'RSI' in manual_df.columns:
        manual_rsi = manual_df['RSI'].dropna()
        print(f"수기 RSI 데이터: {len(manual_rsi)}개")
        print(f"RSI 범위: {manual_rsi.min():.2f} ~ {manual_rsi.max():.2f}")
        
        # RSI 값 비교 (처음 10개)
        print(f"\nRSI 값 비교 (처음 10개):")
        print(f"{'날짜':<12} {'프로그램 RSI':<12} {'수기 RSI':<12} {'차이':<8}")
        print("-" * 50)
        
        for i, date in enumerate(manual_rsi.index[:10]):
            if date in rsi_filtered.index:
                prog_rsi = rsi_filtered.loc[date]
                manual_rsi_val = manual_rsi.loc[date]
                diff = abs(prog_rsi - manual_rsi_val)
                print(f"{date.strftime('%Y-%m'):<12} {prog_rsi:10.2f} {manual_rsi_val:10.2f} {diff:6.2f}")
            else:
                print(f"{date.strftime('%Y-%m'):<12} {'N/A':<10} {manual_rsi.loc[date]:10.2f} {'N/A':<6}")
    
    # 다양한 계산 방법 테스트
    print(f"\n=== 다양한 계산 방법 테스트 ===")
    
    results = {}
    
    # 방법 1: 현재 프로그램 방식 (첫 번째 날짜 가격)
    result1 = calculate_strategy_return(sp500_filtered, rsi_filtered_df, common_periods, strategy_rules, classify_season, initial_capital, method="first")
    results['방법1_첫번째날짜'] = result1
    
    # 방법 2: 마지막 날짜 가격
    result2 = calculate_strategy_return(sp500_filtered, rsi_filtered_df, common_periods, strategy_rules, classify_season, initial_capital, method="last")
    results['방법2_마지막날짜'] = result2
    
    # 방법 3: 평균 가격
    result3 = calculate_strategy_return(sp500_filtered, rsi_filtered_df, common_periods, strategy_rules, classify_season, initial_capital, method="mean")
    results['방법3_평균가격'] = result3
    
    # 방법 4: 수기 작업과 동일한 RSI 사용 (가능한 경우)
    if 'RSI' in manual_df.columns:
        manual_rsi_aligned = align_manual_rsi_data(manual_df, sp500_filtered, common_periods)
        if manual_rsi_aligned is not None:
            result4 = calculate_strategy_with_manual_rsi(sp500_filtered, manual_rsi_aligned, common_periods, strategy_rules, classify_season, initial_capital)
            results['방법4_수기RSI'] = result4
    
    # 결과 비교
    print(f"\n계산 방법별 결과:")
    print(f"{'방법':<20} {'총 수익률':<12} {'수기와 차이':<12} {'CAGR':<8}")
    print("-" * 55)
    
    manual_return = 1139.95
    investment_years = (end_date.year - start_date.year) + (end_date.month - start_date.month) / 12
    
    for method_name, result in results.items():
        total_return = result['total_return']
        cagr = result['cagr']
        diff = abs(manual_return - total_return)
        print(f"{method_name:<20} {total_return:10.2f}% {diff:10.2f}%p {cagr:6.2f}%")
    
    print(f"{'수기_작업':<20} {manual_return:10.2f}% {'0.00':>10}%p {'N/A':>6}")
    
    # 가장 가까운 방법
    closest = min(results.items(), key=lambda x: abs(manual_return - x[1]['total_return']))
    print(f"\n가장 가까운 방법: {closest[0]} (차이: {abs(manual_return - closest[1]['total_return']):.2f}%p)")
    
    return results

def calculate_strategy_return(sp500_filtered, rsi_filtered_df, common_periods, strategy_rules, classify_season, initial_capital, method="first"):
    """전략 수익률을 계산합니다."""
    
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
                
                # 가격 결정 방법
                if method == "first":
                    price = sp500_month[target_style].iloc[0]
                elif method == "last":
                    price = sp500_month[target_style].iloc[-1]
                elif method == "mean":
                    price = sp500_month[target_style].mean()
                else:
                    price = sp500_month[target_style].iloc[0]
                
                if i == 0:
                    current_style = target_style
                    current_shares = portfolio_value / price
                    portfolio_value = current_shares * price
                elif target_style != current_style:
                    # 매도 후 매수
                    if current_style and current_shares > 0:
                        if method == "first":
                            sell_price = sp500_month[current_style].iloc[0]
                        elif method == "last":
                            sell_price = sp500_month[current_style].iloc[-1]
                        elif method == "mean":
                            sell_price = sp500_month[current_style].mean()
                        else:
                            sell_price = sp500_month[current_style].iloc[0]
                            
                        cash = current_shares * sell_price
                        
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                else:
                    portfolio_value = current_shares * price
    
    total_return = (portfolio_value / initial_capital - 1) * 100
    investment_years = len(common_periods) / 12  # 월별 데이터이므로
    cagr = ((portfolio_value / initial_capital) ** (1/investment_years) - 1) * 100 if investment_years > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'final_value': portfolio_value
    }

def align_manual_rsi_data(manual_df, sp500_filtered, common_periods):
    """수기 작업의 RSI 데이터를 정렬합니다."""
    if 'RSI' not in manual_df.columns:
        return None
    
    manual_rsi = manual_df['RSI'].dropna()
    aligned_rsi = []
    
    for period in common_periods:
        # 해당 월의 수기 RSI 찾기
        period_start = pd.Timestamp(period.year, period.month, 1)
        period_end = pd.Timestamp(period.year, period.month, 28) + pd.DateOffset(days=10)
        
        month_rsi = manual_rsi[(manual_rsi.index >= period_start) & (manual_rsi.index <= period_end)]
        
        if len(month_rsi) > 0:
            aligned_rsi.append(month_rsi.iloc[0])
        else:
            aligned_rsi.append(np.nan)
    
    return pd.Series(aligned_rsi, index=range(len(common_periods)))

def calculate_strategy_with_manual_rsi(sp500_filtered, manual_rsi_aligned, common_periods, strategy_rules, classify_season, initial_capital):
    """수기 RSI를 사용한 전략 계산"""
    
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    for i, period in enumerate(common_periods):
        if i >= len(manual_rsi_aligned):
            break
            
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_value = manual_rsi_aligned.iloc[i]
        
        if len(sp500_month) > 0 and pd.notna(rsi_value):
            season = classify_season(rsi_value)
            
            if pd.notna(season):
                target_style = strategy_rules[season]
                price = sp500_month[target_style].iloc[0]  # 첫 번째 날짜 가격
                
                if i == 0:
                    current_style = target_style
                    current_shares = portfolio_value / price
                    portfolio_value = current_shares * price
                elif target_style != current_style:
                    if current_style and current_shares > 0:
                        sell_price = sp500_month[current_style].iloc[0]
                        cash = current_shares * sell_price
                        
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                else:
                    portfolio_value = current_shares * price
    
    total_return = (portfolio_value / initial_capital - 1) * 100
    investment_years = len(common_periods) / 12
    cagr = ((portfolio_value / initial_capital) ** (1/investment_years) - 1) * 100 if investment_years > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'final_value': portfolio_value
    }

if __name__ == "__main__":
    results = analyze_exact_calculation_difference()