import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_manual_data():
    """수기 작업 결과를 로드합니다."""
    try:
        # 11.xlsx 파일 로딩 시도
        df = pd.read_excel('11.xlsx', skiprows=1)  # 첫 행이 제목일 수 있음
        
        # 날짜 컬럼 찾기
        date_col = None
        for col in df.columns:
            if '날짜' in str(col) or 'Date' in str(col) or df[col].dtype == 'datetime64[ns]':
                date_col = col
                break
        
        if date_col is None:
            # 첫 번째 컬럼을 날짜로 가정
            date_col = df.columns[0]
        
        print(f"날짜 컬럼: {date_col}")
        print(f"전체 컬럼: {list(df.columns)}")
        
        # 날짜 컬럼 변환
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"수기 데이터 로딩 완료: {len(df)}개 행")
        print(f"기간: {df.index.min()} ~ {df.index.max()}")
        print(f"컬럼 목록:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        return df
        
    except Exception as e:
        print(f"수기 데이터 로딩 오류: {e}")
        return None

def load_program_data():
    """프로그램에서 사용하는 원본 데이터를 로드합니다."""
    try:
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
        
        print(f"S&P 500 데이터: {len(sp500_df)}개 행")
        print(f"RSI 데이터: {len(rsi_df)}개 행")
        
        return sp500_df, rsi_df
        
    except Exception as e:
        print(f"프로그램 데이터 로딩 오류: {e}")
        return None, None

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

def align_data_detailed(sp500_df, rsi_series, start_date, end_date):
    """데이터 정렬 과정을 상세히 분석합니다."""
    print(f"\n=== 데이터 정렬 상세 분석 ===")
    print(f"요청 기간: {start_date} ~ {end_date}")
    
    # 기간 필터링
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    print(f"필터링 후 S&P 500: {len(sp500_filtered)}개")
    print(f"필터링 후 RSI: {len(rsi_filtered)}개")
    
    # 월별 매칭을 위해 년-월 키 생성
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    # 공통 년-월 찾기
    sp500_periods = set(sp500_filtered['year_month'])
    rsi_periods = set(rsi_filtered_df['year_month'])
    common_periods = sorted(list(sp500_periods.intersection(rsi_periods)))
    
    print(f"S&P 500 월별 기간: {len(sp500_periods)}개")
    print(f"RSI 월별 기간: {len(rsi_periods)}개")
    print(f"공통 기간: {len(common_periods)}개")
    
    # 첫 10개 공통 기간 출력
    print(f"첫 10개 공통 기간: {common_periods[:10]}")
    
    aligned_data = []
    aligned_rsi = []
    aligned_dates = []
    
    for i, period in enumerate(common_periods[:10]):  # 처음 10개만 상세 분석
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            sp500_date = sp500_month.index[0]
            rsi_value = rsi_month['RSI'].iloc[0]
            
            print(f"{period}: S&P500 날짜={sp500_date.strftime('%Y-%m-%d')}, RSI={rsi_value:.2f}")
            
            aligned_dates.append(sp500_date)
            aligned_data.append(sp500_month.drop('year_month', axis=1).iloc[0])
            aligned_rsi.append(rsi_value)
    
    # 전체 처리
    aligned_data = []
    aligned_rsi = []
    aligned_dates = []
    
    for period in common_periods:
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0 and len(rsi_month) > 0:
            sp500_date = sp500_month.index[0]
            rsi_value = rsi_month['RSI'].iloc[0]
            
            aligned_dates.append(sp500_date)
            aligned_data.append(sp500_month.drop('year_month', axis=1).iloc[0])
            aligned_rsi.append(rsi_value)

    sp500_aligned = pd.DataFrame(aligned_data, index=aligned_dates)
    rsi_aligned = pd.Series(aligned_rsi, index=aligned_dates)
    
    print(f"최종 정렬된 데이터: {len(aligned_dates)}개 포인트")
    return sp500_aligned, rsi_aligned

def simulate_manual_strategy(sp500_aligned, rsi_aligned, initial_capital=10000000):
    """수기 전략을 시뮬레이션합니다."""
    
    # 전략 규칙
    strategy_rules = {
        '봄': 'S&P500 Quality',
        '여름': 'S&P500 Momentum', 
        '가을': 'S&P500 Low Volatiltiy Index',
        '겨울': 'S&P500 Low Volatiltiy Index'
    }
    
    # 계절 분류
    seasons = classify_market_season(rsi_aligned)
    
    print(f"\n=== 전략 시뮬레이션 ===")
    print("계절별 분포:")
    season_counts = seasons.value_counts()
    for season, count in season_counts.items():
        print(f"  {season}: {count}회")
    
    portfolio_values = []
    transactions = []
    current_style = None
    current_shares = 0
    cash = 0
    
    print(f"\n처음 10개 월 거래 내역:")
    
    for i, date in enumerate(sp500_aligned.index):
        season = seasons.loc[date] if date in seasons.index else np.nan
        
        if pd.isna(season):
            if i == 0:
                portfolio_value = initial_capital
                cash = initial_capital
            else:
                if current_style and current_shares > 0:
                    portfolio_value = cash + (current_shares * sp500_aligned.loc[date, current_style])
                else:
                    portfolio_value = cash
        else:
            target_style = strategy_rules[season]
            
            if i == 0:
                current_style = target_style
                current_price = sp500_aligned.loc[date, current_style]
                current_shares = initial_capital / current_price
                cash = 0
                portfolio_value = initial_capital
                
                if i < 10:
                    print(f"{date.strftime('%Y-%m')}: {season}({rsi_aligned.loc[date]:.1f}) -> {target_style} @ {current_price:.2f}")
                
            elif target_style != current_style:
                # 매도
                if current_style and current_shares > 0:
                    sell_price = sp500_aligned.loc[date, current_style]
                    cash = current_shares * sell_price
                
                # 매수
                current_style = target_style
                buy_price = sp500_aligned.loc[date, current_style]
                current_shares = cash / buy_price
                cash = 0
                portfolio_value = current_shares * buy_price
                
                if i < 10:
                    print(f"{date.strftime('%Y-%m')}: {season}({rsi_aligned.loc[date]:.1f}) -> {target_style} @ {buy_price:.2f}")
            
            else:
                if current_style and current_shares > 0:
                    portfolio_value = current_shares * sp500_aligned.loc[date, current_style]
                else:
                    portfolio_value = cash
        
        portfolio_values.append(portfolio_value)
    
    portfolio_series = pd.Series(portfolio_values, index=sp500_aligned.index)
    
    # 최종 결과
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    print(f"\n=== 시뮬레이션 결과 ===")
    print(f"초기 자본: {initial_capital:,}원")
    print(f"최종 가치: {final_value:,}원")
    print(f"총 수익률: {total_return:.2f}%")
    
    return portfolio_series, total_return

def compare_data_sources():
    """데이터 소스들을 비교합니다."""
    print("=== 데이터 소스 비교 분석 ===")
    
    # 1. 수기 데이터 로딩
    manual_df = load_manual_data()
    if manual_df is None:
        return
    
    # 2. 프로그램 데이터 로딩
    sp500_df, rsi_df = load_program_data()
    if sp500_df is None or rsi_df is None:
        return
    
    # 3. 1999-2025 기간으로 분석
    start_date = pd.Timestamp(1999, 1, 1)
    end_date = pd.Timestamp(2025, 6, 30)
    
    print(f"\n분석 기간: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")
    
    # 4. 프로그램 방식으로 데이터 정렬 및 시뮬레이션
    rsi_series = rsi_df['RSI'].dropna()
    sp500_aligned, rsi_aligned = align_data_detailed(sp500_df, rsi_series, start_date, end_date)
    
    # 5. 전략 시뮬레이션
    portfolio_series, total_return = simulate_manual_strategy(sp500_aligned, rsi_aligned)
    
    print(f"\n=== 결과 비교 ===")
    print(f"수기 작업 결과: 1139.95%")
    print(f"프로그램 시뮬레이션: {total_return:.2f}%")
    print(f"차이: {1139.95 - total_return:.2f}%p")
    
    # 6. 차이 원인 분석
    print(f"\n=== 차이 원인 분석 ===")
    print(f"1. 데이터 정렬 방식 차이 가능성")
    print(f"2. 월별 매칭 기준 차이 (월말 vs 월초)")
    print(f"3. 가격 적용 시점 차이")
    print(f"4. 복리 계산 방식 차이")
    
    return manual_df, portfolio_series

if __name__ == "__main__":
    manual_df, portfolio_series = compare_data_sources()