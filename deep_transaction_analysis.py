import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_manual_calculation():
    """수기 계산 과정을 상세히 분석합니다."""
    
    print("=== 수기 계산 과정 역추적 분석 ===")
    
    # 데이터 로드
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
    
    manual_df = pd.read_excel('11.xlsx', skiprows=1)
    date_col = manual_df.columns[0]
    manual_df[date_col] = pd.to_datetime(manual_df[date_col])
    manual_df.set_index(date_col, inplace=True)
    manual_df.sort_index(inplace=True)
    
    # 수기 파일에서 사용 가능한 스타일 컬럼들 확인
    print("수기 파일의 스타일 관련 컬럼들:")
    style_columns = {}
    
    # S&P 500 관련 컬럼 찾기
    for col in manual_df.columns:
        col_str = str(col).upper()
        if 'S&P500' in col_str or 'S&P 500' in col_str:
            if 'MOMENTUM' in col_str or '모멘텀' in col_str:
                style_columns['Momentum'] = col
            elif 'QUALITY' in col_str or '퀄리티' in col_str:
                style_columns['Quality'] = col
            elif 'LOW VOL' in col_str or '로볼' in col_str or 'VOLATILTIY' in col_str:
                style_columns['Low Volatility'] = col
            elif 'VALUE' in col_str or '가치' in col_str:
                style_columns['Value'] = col
            elif 'GROWTH' in col_str or '성장' in col_str:
                style_columns['Growth'] = col
            elif 'DIV' in col_str or '배당' in col_str:
                style_columns['Dividend'] = col
    
    print("매핑된 스타일 컬럼들:")
    for style, col in style_columns.items():
        print(f"  {style}: {col}")
    
    # 수기 전략 시뮬레이션 (여러 가능성 테스트)
    initial_capital = 10000000
    start_date = pd.Timestamp(1999, 1, 1)
    end_date = pd.Timestamp(2025, 6, 30)
    
    # 가능한 매핑들 테스트
    possible_mappings = [
        # 매핑 1: 프로그램과 동일한 매핑 시도
        {
            '봄': 'S&P 500 퀄리티',  # 수기 파일의 정확한 컬럼명 사용
            '여름': 'S&P500 모멘텀',
            '가을': 'S&P 로볼',
            '겨울': 'S&P 로볼'
        },
        # 매핑 2: 다른 컬럼 변형들
        {
            '봄': 'S&P500 Quality',
            '여름': 'S&P500 Momentum', 
            '가을': 'S&P500 Low Volatiltiy Index',
            '겨울': 'S&P500 Low Volatiltiy Index'
        }
    ]
    
    # 수기 파일에서 실제 사용 가능한 컬럼들로 매핑 재구성
    if len(style_columns) > 0:
        # 수기 파일 기준으로 매핑 생성
        manual_mapping = {}
        
        # Quality 찾기
        quality_col = None
        for col in manual_df.columns:
            if '퀄리티' in str(col) or 'Quality' in str(col):
                quality_col = col
                break
        
        # Momentum 찾기  
        momentum_col = None
        for col in manual_df.columns:
            if '모멘텀' in str(col) or 'Momentum' in str(col):
                momentum_col = col
                break
        
        # Low Volatility 찾기
        lowvol_col = None
        for col in manual_df.columns:
            if '로볼' in str(col) or 'Low Vol' in str(col) or 'Volatiltiy' in str(col):
                lowvol_col = col
                break
        
        if quality_col and momentum_col and lowvol_col:
            manual_mapping = {
                '봄': quality_col,
                '여름': momentum_col,
                '가을': lowvol_col,
                '겨울': lowvol_col
            }
            
            print(f"\n수기 파일 기반 매핑:")
            for season, col in manual_mapping.items():
                print(f"  {season}: {col}")
            
            # 수기 데이터로 전략 실행
            result = simulate_strategy_with_manual_data(manual_df, manual_mapping, initial_capital)
            
            print(f"\n수기 데이터 시뮬레이션 결과:")
            print(f"총 수익률: {result['total_return']:.2f}%")
            print(f"최종 가치: {result['final_value']:,.0f}원")
            
            # 1139.95%와의 차이 분석
            target_return = 1139.95
            diff = abs(result['total_return'] - target_return)
            print(f"목표 수익률과 차이: {diff:.2f}%p")
            
            if diff < 50:  # 차이가 50%p 미만이면 거의 일치
                print("✓ 수기 계산과 거의 일치하는 결과!")
            else:
                print("✗ 여전히 차이가 큼. 다른 계산 방식 존재 가능성")
    
    # 복리 계산 방식 차이 가능성 검토
    print(f"\n=== 복리 계산 방식 차이 검토 ===")
    
    # 단순 복리 vs 연속 복리 차이
    simple_compound = (1 + 1027.58/100)
    target_compound = (1 + 1139.95/100)
    
    ratio = target_compound / simple_compound
    print(f"수기/프로그램 비율: {ratio:.4f}")
    print(f"이는 약 {(ratio-1)*100:.1f}% 추가 수익률에 해당")
    
    # 월별 리밸런싱 vs 연간 리밸런싱 차이 가능성
    print(f"\n=== 리밸런싱 주기 차이 가능성 ===")
    
    # 수기 작업에서 거래 횟수 분석
    if 'RSI' in manual_df.columns:
        manual_rsi = manual_df['RSI'].dropna()
        seasons = []
        for rsi_val in manual_rsi:
            if rsi_val >= 70:
                seasons.append('여름')
            elif rsi_val >= 50:
                seasons.append('봄') 
            elif rsi_val >= 30:
                seasons.append('가을')
            else:
                seasons.append('겨울')
        
        # 계절 변화 횟수 계산
        season_changes = 0
        for i in range(1, len(seasons)):
            if seasons[i] != seasons[i-1]:
                season_changes += 1
        
        print(f"수기 데이터 기준 계절 변화: {season_changes}회")
        print(f"이론적 거래 횟수: {season_changes + 1}회 (초기 투자 포함)")

def simulate_strategy_with_manual_data(manual_df, strategy_mapping, initial_capital):
    """수기 데이터를 직접 사용한 전략 시뮬레이션"""
    
    if 'RSI' not in manual_df.columns:
        return {'total_return': 0, 'final_value': initial_capital}
    
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    manual_rsi = manual_df['RSI'].dropna()
    
    print(f"\n수기 데이터 전략 실행 (처음 10개월):")
    print(f"{'날짜':<12} {'RSI':<6} {'계절':<6} {'스타일':<20} {'가격':<10} {'포트폴리오':<15}")
    print("-" * 80)
    
    for i, (date, rsi_value) in enumerate(manual_rsi.items()):
        # 계절 분류
        if rsi_value >= 70:
            season = '여름'
        elif rsi_value >= 50:
            season = '봄'
        elif rsi_value >= 30:
            season = '가을'
        else:
            season = '겨울'
        
        target_style = strategy_mapping[season]
        
        # 해당 날짜의 가격 확인
        if target_style in manual_df.columns and date in manual_df.index:
            price = manual_df.loc[date, target_style]
            
            if pd.notna(price) and price > 0:
                if i == 0:
                    # 초기 투자
                    current_style = target_style
                    current_shares = portfolio_value / price
                    portfolio_value = current_shares * price
                elif target_style != current_style:
                    # 매도 후 매수
                    if current_style and current_shares > 0 and current_style in manual_df.columns:
                        if date in manual_df.index:
                            sell_price = manual_df.loc[date, current_style]
                            if pd.notna(sell_price) and sell_price > 0:
                                cash = current_shares * sell_price
                                
                                current_style = target_style
                                current_shares = cash / price
                                portfolio_value = current_shares * price
                else:
                    # 동일 스타일 유지
                    portfolio_value = current_shares * price
                
                # 처음 10개월 출력
                if i < 10:
                    print(f"{date.strftime('%Y-%m'):<12} {rsi_value:5.1f} {season:<6} {target_style[:20]:<20} {price:9.2f} {portfolio_value:13,.0f}")
    
    total_return = (portfolio_value / initial_capital - 1) * 100
    
    return {
        'total_return': total_return,
        'final_value': portfolio_value
    }

def check_data_precision_differences():
    """데이터 정밀도 차이를 확인합니다."""
    
    print(f"\n=== 데이터 정밀도 차이 확인 ===")
    
    # 프로그램 데이터
    sp500_df = pd.read_excel('sp500_data.xlsx')
    date_column = sp500_df.columns[0] 
    sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
    sp500_df.set_index(date_column, inplace=True)
    
    # 수기 데이터
    manual_df = pd.read_excel('11.xlsx', skiprows=1)
    date_col = manual_df.columns[0]
    manual_df[date_col] = pd.to_datetime(manual_df[date_col])
    manual_df.set_index(date_col, inplace=True)
    
    # 공통 날짜에서 가격 비교
    common_dates = sp500_df.index.intersection(manual_df.index)[:10]
    
    print(f"가격 데이터 정밀도 비교 (처음 10개 날짜):")
    print(f"{'날짜':<12} {'프로그램':<12} {'수기':<12} {'차이%':<8}")
    print("-" * 50)
    
    # 모멘텀 지수 비교 (있는 경우)
    prog_momentum_col = 'S&P500 Momentum'
    manual_momentum_col = None
    
    for col in manual_df.columns:
        if '모멘텀' in str(col) or 'Momentum' in str(col):
            manual_momentum_col = col
            break
    
    if manual_momentum_col and prog_momentum_col in sp500_df.columns:
        for date in common_dates:
            prog_price = sp500_df.loc[date, prog_momentum_col]
            manual_price = manual_df.loc[date, manual_momentum_col]
            
            if pd.notna(prog_price) and pd.notna(manual_price) and manual_price != 0:
                diff_pct = ((manual_price - prog_price) / prog_price) * 100
                print(f"{date.strftime('%Y-%m'):<12} {prog_price:10.4f} {manual_price:10.4f} {diff_pct:6.2f}%")

if __name__ == "__main__":
    load_and_analyze_manual_calculation()
    check_data_precision_differences()