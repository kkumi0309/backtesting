import pandas as pd
import numpy as np
import warnings
import sys
sys.path.append('.')
from AA import load_sp500_data, load_rsi_data, align_data, classify_market_season, run_rotation_strategy

warnings.filterwarnings('ignore')

def test_fixed_momentum_only():
    """수정된 AA.py로 모멘텀 온리 전략 테스트"""
    
    print("=== 수정된 AA.py 모멘텀 온리 전략 테스트 ===")
    
    # 데이터 로드 (절대 경로 사용)
    import os
    base_path = os.getcwd()
    
    sp500_df = load_sp500_data(os.path.join(base_path, 'sp500_data.xlsx'))
    if sp500_df is None:
        print("sp500_data.xlsx 로딩 실패")
        return None, None, 0
        
    rsi_series = load_rsi_data(os.path.join(base_path, 'RSI_DATE.xlsx'))
    if rsi_series is None:
        print("RSI_DATE.xlsx 로딩 실패")
        return None, None, 0
    
    # 기간 설정 (1999-01 ~ 2001-07)
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2001-07-31')
    
    print(f"백테스팅 기간: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")
    
    # 데이터 정렬
    sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
    
    # 계절 분류
    seasons = classify_market_season(rsi_aligned)
    
    print(f"계절 분포: {seasons.value_counts().to_dict()}")
    
    # 모멘텀 온리 전략
    momentum_strategy = {
        '여름': 'S&P500 Momentum',
        '봄': 'S&P500 Momentum',
        '가을': 'S&P500 Momentum',
        '겨울': 'S&P500 Momentum'
    }
    
    # 초기 투자금
    initial_capital = 10000000  # 1천만원
    
    # 백테스팅 실행
    portfolio_series, transactions, season_stats = run_rotation_strategy(
        sp500_aligned, "모멘텀 온리", momentum_strategy, seasons, initial_capital
    )
    
    # 결과 계산
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    print(f"\n=== 백테스팅 결과 ===")
    print(f"초기 투자금: {initial_capital:,}원")
    print(f"최종 가치: {final_value:,.0f}원")
    print(f"총 수익률: {total_return:.2%}")
    
    # 상세 결과
    print(f"\n=== 상세 정보 ===")
    print(f"시작일: {portfolio_series.index[0].strftime('%Y-%m-%d')}")
    print(f"종료일: {portfolio_series.index[-1].strftime('%Y-%m-%d')}")
    
    # 거래 내역
    print(f"\n=== 거래 내역 ===")
    for i, tx in enumerate(transactions):
        print(f"{i+1}. {tx['date'].strftime('%Y-%m-%d')}: {tx['action']} - {tx.get('to_style', 'N/A')} ({tx['value']:,.0f}원)")
    
    # 모멘텀 지수 값 변화
    momentum_start = sp500_aligned.iloc[0]['S&P500 Momentum']
    momentum_end = sp500_aligned.iloc[-1]['S&P500 Momentum']
    momentum_return = (momentum_end / momentum_start) - 1
    
    print(f"\nS&P500 Momentum 지수:")
    print(f"시작 값: {momentum_start:.3f}")
    print(f"종료 값: {momentum_end:.3f}")
    print(f"지수 수익률: {momentum_return:.2%}")
    
    # 월별 포트폴리오 가치 출력 (처음 5개와 마지막 5개)
    print(f"\n=== 월별 포트폴리오 가치 (처음 5개) ===")
    for i, (date, value) in enumerate(portfolio_series.head().items()):
        season = seasons.loc[date] if date in seasons.index else 'N/A'
        rsi_val = rsi_aligned.loc[date] if date in rsi_aligned.index else 'N/A'
        momentum_val = sp500_aligned.loc[date]['S&P500 Momentum']
        print(f"{date.strftime('%Y-%m')}: {value:,.0f}원 (RSI:{rsi_val}, 계절:{season}, 모멘텀:{momentum_val:.3f})")
    
    print(f"\n=== 월별 포트폴리오 가치 (마지막 5개) ===")
    for i, (date, value) in enumerate(portfolio_series.tail().items()):
        season = seasons.loc[date] if date in seasons.index else 'N/A'
        rsi_val = rsi_aligned.loc[date] if date in rsi_aligned.index else 'N/A'
        momentum_val = sp500_aligned.loc[date]['S&P500 Momentum']
        print(f"{date.strftime('%Y-%m')}: {value:,.0f}원 (RSI:{rsi_val}, 계절:{season}, 모멘텀:{momentum_val:.3f})")
    
    return portfolio_series, total_return, len(portfolio_series)

if __name__ == "__main__":
    portfolio_series, total_return, data_points = test_fixed_momentum_only()
    print(f"\n[결과 요약] 데이터 포인트: {data_points}개, 총 수익률: {total_return:.2%}")