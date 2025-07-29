import pandas as pd
import numpy as np
import warnings
import sys
import os
sys.path.append('.')

warnings.filterwarnings('ignore')

# AA.py에서 함수들 직접 임포트
from AA import load_sp500_data, load_rsi_data, align_data, classify_market_season, run_rotation_strategy

def debug_aa_momentum():
    """AA.py의 모멘텀 온리 전략 디버깅"""
    
    print("=== AA.py 모멘텀 온리 전략 디버깅 ===")
    
    # 스크립트 디렉토리로 경로 설정 (AA.py와 동일)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터 로딩
    sp500_df = load_sp500_data(os.path.join(script_dir, 'sp500_data.xlsx'))
    rsi_series = load_rsi_data(os.path.join(script_dir, 'RSI_DATE.xlsx'))
    
    # 기간 설정 (1999-01 ~ 2001-07)
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2001-07-31')
    
    print(f"백테스팅 기간: {start_date.strftime('%Y-%m')} ~ {end_date.strftime('%Y-%m')}")
    
    # 데이터 정렬
    sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
    
    print(f"정렬된 데이터 기간: {sp500_aligned.index[0].strftime('%Y-%m-%d')} ~ {sp500_aligned.index[-1].strftime('%Y-%m-%d')}")
    print(f"데이터 포인트: {len(sp500_aligned)}개")
    
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
    
    initial_capital = 10000000
    
    # 백테스팅 실행
    portfolio_series, transactions, season_stats = run_rotation_strategy(
        sp500_aligned, "모멘텀 온리", momentum_strategy, seasons, initial_capital
    )
    
    # 결과 분석
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    print(f"\n=== 백테스팅 결과 ===")
    print(f"초기 투자금: {initial_capital:,}원")
    print(f"최종 가치: {final_value:,.0f}원")
    print(f"총 수익률: {total_return:.2%}")
    
    # 거래 내역
    print(f"\n=== 거래 내역 ({len(transactions)}건) ===")
    for i, tx in enumerate(transactions):
        print(f"{i+1}. {tx['date'].strftime('%Y-%m-%d')}: {tx['action']} - {tx.get('to_style', 'N/A')} ({tx['value']:,.0f}원)")
    
    # 모멘텀 지수 값 변화
    momentum_start = sp500_aligned.iloc[0]['S&P500 Momentum'] 
    momentum_end = sp500_aligned.iloc[-1]['S&P500 Momentum']
    momentum_return = (momentum_end / momentum_start) - 1
    
    print(f"\n=== S&P500 Momentum 지수 변화 ===")
    print(f"시작 값: {momentum_start:.3f}")
    print(f"종료 값: {momentum_end:.3f}")
    print(f"지수 수익률: {momentum_return:.4%}")
    
    # 포트폴리오와 지수 수익률 비교
    portfolio_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
    print(f"\n=== 수익률 비교 ===")
    print(f"포트폴리오 수익률: {portfolio_return:.4%}")
    print(f"모멘텀 지수 수익률: {momentum_return:.4%}")
    print(f"차이: {(portfolio_return - momentum_return):.6%}")
    
    # Buy & Hold 최적화가 적용되었는지 확인
    unique_styles = set(momentum_strategy.values())
    is_buy_and_hold = len(unique_styles) == 1
    print(f"\nBuy & Hold 최적화 적용됨: {is_buy_and_hold}")
    
    # 상세 월별 데이터 (처음 5개, 마지막 5개)
    print(f"\n=== 상세 월별 데이터 (처음 5개) ===")
    for i in range(min(5, len(portfolio_series))):
        date = portfolio_series.index[i]
        value = portfolio_series.iloc[i]
        momentum_val = sp500_aligned.loc[date]['S&P500 Momentum']
        rsi_val = rsi_aligned.loc[date] if date in rsi_aligned.index else 'N/A'
        season = seasons.loc[date] if date in seasons.index else 'N/A'
        
        print(f"{date.strftime('%Y-%m')}: {value:,.0f}원 (모멘텀:{momentum_val:.3f}, RSI:{rsi_val}, 계절:{season})")
    
    print(f"\n=== 상세 월별 데이터 (마지막 5개) ===")
    for i in range(max(0, len(portfolio_series)-5), len(portfolio_series)):
        date = portfolio_series.index[i]
        value = portfolio_series.iloc[i]
        momentum_val = sp500_aligned.loc[date]['S&P500 Momentum']
        rsi_val = rsi_aligned.loc[date] if date in rsi_aligned.index else 'N/A'
        season = seasons.loc[date] if date in seasons.index else 'N/A'
        
        print(f"{date.strftime('%Y-%m')}: {value:,.0f}원 (모멘텀:{momentum_val:.3f}, RSI:{rsi_val}, 계절:{season})")
    
    return portfolio_series, total_return

if __name__ == "__main__":
    portfolio_series, total_return = debug_aa_momentum()