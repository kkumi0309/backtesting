import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_sp500_data(file_path):
    """S&P 500 스타일 지수 데이터를 로드합니다."""
    try:
        df = pd.read_excel(file_path)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"S&P 500 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        return df
    except Exception as e:
        print(f"S&P 500 데이터 로딩 오류: {e}")
        return None

def load_rsi_data(file_path):
    """기존에 계산된 RSI 데이터를 로드합니다."""
    try:
        df = pd.read_excel(file_path, skiprows=1)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        rsi_series = df['RSI'].dropna()
        print(f"RSI 데이터 로딩 완료: {len(rsi_series)}개 데이터 포인트")
        return rsi_series
    except Exception as e:
        print(f"RSI 데이터 로딩 오류: {e}")
        return None

def classify_market_season(rsi):
    """RSI 값을 기반으로 시장 계절을 분류합니다."""
    def get_season(rsi_value):
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
    
    return rsi.apply(get_season)

def align_data_corrected(sp500_df, rsi_series, start_date, end_date):
    """수정된 데이터 정렬 함수 - 더 정확한 매칭"""
    
    # 기간 필터링
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    print(f"필터링 후 - S&P 500: {len(sp500_filtered)}개, RSI: {len(rsi_filtered)}개")
    
    # 월별 매칭을 위해 년-월 키 생성
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    # 공통 년-월 찾기
    sp500_periods = sorted(list(set(sp500_filtered['year_month'])))
    rsi_periods = sorted(list(set(rsi_filtered_df['year_month'])))
    common_periods = sorted(list(set(sp500_periods).intersection(set(rsi_periods))))
    
    print(f"공통 기간: {len(common_periods)}개")
    
    aligned_data = []
    aligned_rsi = []
    aligned_dates = []
    
    for period in common_periods:
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            # 중요: 해당 월의 첫 번째 데이터 사용 (기존과 동일)
            sp500_date = sp500_month.index[0]
            rsi_value = rsi_month['RSI'].iloc[0]
            
            aligned_dates.append(sp500_date)
            aligned_data.append(sp500_month.drop('year_month', axis=1).iloc[0])
            aligned_rsi.append(rsi_value)

    if not aligned_dates:
        raise ValueError("데이터 정렬 후 남은 데이터가 없습니다.")

    sp500_aligned = pd.DataFrame(aligned_data, index=aligned_dates)
    rsi_aligned = pd.Series(aligned_rsi, index=aligned_dates)
    
    print(f"최종 정렬 완료: {len(aligned_dates)}개 데이터 포인트")
    return sp500_aligned, rsi_aligned

def run_corrected_backtest(sp500_aligned, rsi_aligned, strategy_rules, initial_capital=10000000):
    """수정된 백테스팅 - 정밀도와 계산 방식 개선"""
    
    print(f"\n=== 수정된 백테스팅 실행 ===")
    
    # 고정밀도를 위한 Decimal 사용
    from decimal import Decimal, getcontext
    getcontext().prec = 28  # 28자리 정밀도
    
    portfolio_value = Decimal(str(initial_capital))
    current_shares = Decimal('0')
    current_style = None
    cash = Decimal('0')
    
    # 계절 분류
    seasons = classify_market_season(rsi_aligned)
    
    transactions = []
    portfolio_values = []
    season_stats = {'여름': [], '봄': [], '가을': [], '겨울': []}
    
    print(f"거래 내역 (처음 10개):")
    print(f"{'날짜':<12} {'RSI':<6} {'계절':<6} {'스타일':<20} {'가격':<12} {'포트폴리오':<15}")
    print("-" * 85)
    
    for i, date in enumerate(sp500_aligned.index):
        season = seasons.loc[date] if date in seasons.index else np.nan
        
        if pd.isna(season):
            if i == 0:
                portfolio_value = Decimal(str(initial_capital))
                cash = portfolio_value
            else:
                if current_style and current_shares > 0:
                    current_price = Decimal(str(sp500_aligned.loc[date, current_style]))
                    portfolio_value = current_shares * current_price + cash
                else:
                    portfolio_value = cash
        else:
            target_style = strategy_rules[season]
            current_price = Decimal(str(sp500_aligned.loc[date, target_style]))
            
            if i == 0:
                # 초기 투자
                current_style = target_style
                current_shares = portfolio_value / current_price
                cash = Decimal('0')
                portfolio_value = current_shares * current_price
                
                transactions.append({
                    'date': date,
                    'season': season,
                    'action': '초기투자',
                    'style': current_style,
                    'price': float(current_price),
                    'shares': float(current_shares),
                    'value': float(portfolio_value)
                })
                
            elif target_style != current_style:
                # 스타일 변경 - 매도 후 매수
                if current_style and current_shares > 0:
                    sell_price = Decimal(str(sp500_aligned.loc[date, current_style]))
                    cash = current_shares * sell_price
                    
                    transactions.append({
                        'date': date,
                        'season': season,
                        'action': '매도',
                        'style': current_style,
                        'price': float(sell_price),
                        'shares': float(current_shares),
                        'value': float(cash)
                    })
                
                # 새 스타일 매수
                current_style = target_style
                current_shares = cash / current_price
                cash = Decimal('0')
                portfolio_value = current_shares * current_price
                
                transactions.append({
                    'date': date,
                    'season': season,
                    'action': '매수',
                    'style': current_style,
                    'price': float(current_price),
                    'shares': float(current_shares),
                    'value': float(portfolio_value)
                })
                
            else:
                # 동일 스타일 유지
                portfolio_value = current_shares * current_price + cash
            
            # 수익률 계산 (전월 대비)
            if i > 0:
                prev_value = portfolio_values[-1] if portfolio_values else Decimal(str(initial_capital))
                period_return = float((portfolio_value / prev_value) - 1)
                season_stats[season].append(period_return)
        
        portfolio_values.append(portfolio_value)
        
        # 처음 10개 거래 출력
        if i < 10:
            rsi_val = rsi_aligned.loc[date] if date in rsi_aligned.index else 0
            date_str = date.strftime('%Y-%m')
            season_str = season if pd.notna(season) else 'N/A'
            style_str = target_style[:20] if pd.notna(season) else 'N/A'
            price_val = float(current_price) if pd.notna(season) else 0
            portfolio_val = float(portfolio_value)
            
            print(f"{date_str:<12} {rsi_val:5.1f} {season_str:<6} {style_str:<20} {price_val:10.2f} {portfolio_val:13,.0f}")
    
    # 최종 결과 계산
    final_value = float(portfolio_values[-1])
    total_return = (final_value / initial_capital - 1) * 100
    
    # 연평균 수익률 (CAGR)
    start_date = sp500_aligned.index[0]
    end_date = sp500_aligned.index[-1]
    investment_years = (end_date - start_date).days / 365.25
    cagr = (final_value / initial_capital) ** (1 / investment_years) - 1 if investment_years > 0 else 0
    
    # 최대 낙폭 (MDD)
    portfolio_series = pd.Series([float(pv) for pv in portfolio_values], index=sp500_aligned.index)
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max) - 1
    mdd = drawdown.min()
    
    # 변동성
    returns = portfolio_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(12)
    
    # 샤프 비율
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'final_value': final_value,
        'cagr': cagr * 100,
        'mdd': mdd,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'investment_years': investment_years,
        'transactions': transactions,
        'season_stats': season_stats,
        'portfolio_series': portfolio_series
    }

def main():
    """메인 실행 함수"""
    print("=== 수정된 백테스팅 실행 ===")
    
    # 1. 데이터 로딩
    sp500_df = load_sp500_data('sp500_data.xlsx')
    if sp500_df is None:
        return
    
    rsi_series = load_rsi_data('RSI_DATE.xlsx')
    if rsi_series is None:
        return
    
    # 2. 백테스팅 설정
    strategy_rules = {
        '봄': 'S&P500 Quality',
        '여름': 'S&P500 Momentum', 
        '가을': 'S&P500 Low Volatiltiy Index',
        '겨울': 'S&P500 Low Volatiltiy Index'
    }
    
    start_date = pd.Timestamp(1999, 1, 1)
    end_date = pd.Timestamp(2025, 6, 30)
    initial_capital = 10000000
    
    print(f"\n전략 구성:")
    for season, style in strategy_rules.items():
        print(f"  {season}: {style}")
    
    print(f"\n백테스팅 기간: {start_date.strftime('%Y년 %m월')} ~ {end_date.strftime('%Y년 %m월')}")
    print(f"초기 투자금: {initial_capital:,}원")
    
    try:
        # 3. 데이터 정렬
        sp500_aligned, rsi_aligned = align_data_corrected(sp500_df, rsi_series, start_date, end_date)
        
        # 4. 수정된 백테스팅 실행
        result = run_corrected_backtest(sp500_aligned, rsi_aligned, strategy_rules, initial_capital)
        
        # 5. 결과 출력
        print(f"\n" + "="*80)
        print(f"           수정된 백테스팅 결과")
        print("="*80)
        
        print(f"투자 기간: {result['investment_years']:.1f}년")
        print(f"총 수익률: {result['total_return']:+.2f}%")
        print(f"연평균 수익률 (CAGR): {result['cagr']:+.2f}%")
        print(f"최대 낙폭 (MDD): {result['mdd']:.2%}")
        print(f"변동성: {result['volatility']:.2%}")
        print(f"샤프 비율: {result['sharpe_ratio']:.2f}")
        print(f"최종 가치: {result['final_value']:,.0f}원")
        
        # 6. 수기 계산과 비교
        manual_return = 1139.95
        program_return = result['total_return']
        difference = abs(manual_return - program_return)
        
        print(f"\n" + "="*50)
        print(f"수기 계산과의 비교")
        print("="*50)
        print(f"수기 계산 결과:   {manual_return:8.2f}%")
        print(f"수정된 프로그램:  {program_return:8.2f}%")
        print(f"차이:            {difference:8.2f}%p")
        
        if difference < 10:
            print("✓ 수기 계산과 거의 일치!")
            success_status = "성공"
        elif difference < 50:
            print("△ 상당히 개선됨")
            success_status = "개선"
        else:
            print("✗ 여전히 큰 차이")
            success_status = "실패"
        
        # 7. 거래 내역 요약
        transactions = result['transactions']
        print(f"\n거래 내역 요약:")
        print(f"총 거래 횟수: {len(transactions)}회")
        
        buy_count = len([t for t in transactions if t['action'] in ['매수', '초기투자']])
        sell_count = len([t for t in transactions if t['action'] == '매도'])
        print(f"매수 거래: {buy_count}회")
        print(f"매도 거래: {sell_count}회")
        
        print(f"\n[{success_status.upper()}] 수정된 백테스팅 완료!")
        
        return result
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()