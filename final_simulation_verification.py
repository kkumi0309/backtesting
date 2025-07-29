import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def verify_with_simulation_file():
    """시뮬레이션 파일로 백테스팅을 검증합니다."""
    
    print("=== S&P 시뮬레이션 파일 백테스팅 검증 ===")
    
    try:
        # 시뮬레이션 파일 로딩
        xl_file = pd.ExcelFile('S&P 시뮬레이션.xlsx')
        sheet_name = xl_file.sheet_names[0]
        
        # header=1로 읽기 (RSI 데이터가 여기 있었음)
        df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name, header=1)
        
        print(f"데이터 로딩 완료: {df.shape}")
        print(f"컬럼들: {list(df.columns)}")
        
        # 날짜 인덱스 설정
        date_col = df.columns[0]  # '날짜'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        print(f"기간: {df.index.min()} ~ {df.index.max()}")
        print(f"데이터 포인트: {len(df)}개")
        
        # RSI 데이터 확인
        rsi_data = pd.to_numeric(df['RSI'], errors='coerce').dropna()
        print(f"\nRSI 데이터: {len(rsi_data)}개")
        print(f"RSI 범위: {rsi_data.min():.2f} ~ {rsi_data.max():.2f}")
        
        # 스타일 지수 컬럼 매핑
        style_mapping = {
            '봄': '퀄리티',        # Quality
            '여름': '모멘텀',       # Momentum  
            '가을': 'S&P 로볼',    # Low Volatility
            '겨울': 'S&P 로볼'     # Low Volatility
        }
        
        print(f"\n전략 매핑:")
        for season, style in style_mapping.items():
            if style in df.columns:
                print(f"  {season}: {style} ✓")
            else:
                print(f"  {season}: {style} ✗ (컬럼 없음)")
        
        # 백테스팅 실행
        result = run_backtest_with_simulation_data(df, rsi_data, style_mapping)
        
        if result:
            print(f"\n=== 시뮬레이션 파일 백테스팅 결과 ===")
            print(f"총 수익률: {result['total_return']:.2f}%")
            print(f"최종 가치: {result['final_value']:,.0f}원")
            
            # 기존 결과들과 비교
            manual_return = 1139.95
            program_return = 1027.58
            sim_return = result['total_return']
            
            print(f"\n=== 결과 비교 ===")
            print(f"{'구분':<15} {'수익률':<12} {'차이(vs수기)':<15}")
            print("-" * 45)
            print(f"{'수기 계산':<15} {manual_return:>10.2f}% {0:>13.2f}%p")
            print(f"{'기존 프로그램':<15} {program_return:>10.2f}% {manual_return-program_return:>13.2f}%p")
            print(f"{'시뮬레이션검증':<15} {sim_return:>10.2f}% {manual_return-sim_return:>13.2f}%p")
            
            # 가장 가까운 결과 확인
            diffs = {
                '기존 프로그램': abs(manual_return - program_return),
                '시뮬레이션검증': abs(manual_return - sim_return)
            }
            
            closest = min(diffs.items(), key=lambda x: x[1])
            print(f"\n수기 계산에 가장 가까운 결과: {closest[0]} (차이 {closest[1]:.2f}%p)")
            
            return result
        
    except Exception as e:
        print(f"검증 오류: {e}")
        return None

def run_backtest_with_simulation_data(df, rsi_data, style_mapping):
    """시뮬레이션 데이터로 백테스팅을 실행합니다."""
    
    initial_capital = 10000000
    portfolio_value = initial_capital
    current_shares = 0
    current_style = None
    
    transactions = []
    
    print(f"\n백테스팅 실행 (처음 10개월):")
    print(f"{'날짜':<12} {'RSI':<6} {'계절':<6} {'스타일':<15} {'가격':<10} {'포트폴리오':<15}")
    print("-" * 75)
    
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
        
        target_style = style_mapping[season]
        
        # 해당 스타일의 가격 확인
        if target_style not in df.columns or date not in df.index:
            continue
        
        try:
            price = pd.to_numeric(df.loc[date, target_style], errors='coerce')
            if pd.isna(price) or price <= 0:
                continue
        except:
            continue
        
        # 거래 로직
        if i == 0:
            # 초기 투자
            current_style = target_style
            current_shares = portfolio_value / price
            portfolio_value = current_shares * price
            
            transactions.append({
                'date': date,
                'action': '초기투자',
                'style': current_style,
                'price': price,
                'shares': current_shares,
                'value': portfolio_value
            })
            
        elif target_style != current_style:
            # 스타일 변경 - 매도 후 매수
            if current_style and current_shares > 0 and current_style in df.columns:
                try:
                    sell_price = pd.to_numeric(df.loc[date, current_style], errors='coerce')
                    if pd.notna(sell_price) and sell_price > 0:
                        cash = current_shares * sell_price
                        
                        # 새 스타일 매수
                        current_style = target_style
                        current_shares = cash / price
                        portfolio_value = current_shares * price
                        
                        transactions.append({
                            'date': date,
                            'action': '리밸런싱',
                            'style': current_style,
                            'price': price,
                            'shares': current_shares,
                            'value': portfolio_value
                        })
                except:
                    pass
        else:
            # 동일 스타일 유지
            portfolio_value = current_shares * price
        
        # 처음 10개월 출력
        if i < 10:
            date_str = date.strftime('%Y-%m') if hasattr(date, 'strftime') else str(date)[:7]
            print(f"{date_str:<12} {rsi_value:5.1f} {season:<6} {target_style:<15} {price:9.2f} {portfolio_value:13,.0f}")
    
    # 최종 결과 계산
    total_return = ((portfolio_value / initial_capital) - 1) * 100
    
    # 거래 통계
    print(f"\n거래 통계:")
    print(f"총 거래 횟수: {len(transactions)}회")
    
    # 계절별 거래 분석
    season_trades = {}
    for trans in transactions:
        if 'date' in trans:
            date = trans['date']
            if date in rsi_data.index:
                rsi_val = rsi_data.loc[date]
                if rsi_val >= 70:
                    season = '여름'
                elif rsi_val >= 50:
                    season = '봄'
                elif rsi_val >= 30:
                    season = '가을'
                else:
                    season = '겨울'
                
                season_trades[season] = season_trades.get(season, 0) + 1
    
    print("계절별 거래 횟수:")
    for season in ['봄', '여름', '가을', '겨울']:
        count = season_trades.get(season, 0)
        print(f"  {season}: {count}회")
    
    return {
        'total_return': total_return,
        'final_value': portfolio_value,
        'transactions': transactions,
        'season_trades': season_trades
    }

def find_exact_calculation_method():
    """정확한 계산 방식을 찾기 위한 추가 분석"""
    
    print(f"\n=== 정확한 계산 방식 탐색 ===")
    
    try:
        # 시뮬레이션 파일에서 다른 수익률 컬럼들 확인
        xl_file = pd.ExcelFile('S&P 시뮬레이션.xlsx')
        sheet_name = xl_file.sheet_names[0]
        df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name, header=1)
        
        # 날짜 설정
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # 수익률 관련 컬럼들 분석
        return_columns = [col for col in df.columns if '수익률' in str(col)]
        
        print("수익률 관련 컬럼들:")
        target_return = 1139.95
        
        for col in return_columns:
            try:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 0:
                    # 다양한 해석 방법
                    interpretations = []
                    
                    # 방법 1: 최종값이 누적 수익률
                    final_val = data.iloc[-1]
                    if 100 < final_val < 5000:  # 합리적 범위
                        interpretations.append(('최종값', final_val))
                    
                    # 방법 2: 초기 대비 배수
                    if len(data) > 1:
                        initial_val = data.iloc[0]
                        if initial_val > 0:
                            ratio_return = ((data.iloc[-1] / initial_val) - 1) * 100
                            if 100 < ratio_return < 5000:
                                interpretations.append(('비율계산', ratio_return))
                    
                    # 방법 3: 평균값 기준
                    mean_val = data.mean()
                    if 100 < mean_val < 5000:
                        interpretations.append(('평균값', mean_val))
                    
                    print(f"\n  {col}:")
                    for method, value in interpretations:
                        diff = abs(value - target_return)
                        print(f"    {method}: {value:.2f}% (차이: {diff:.2f}%p)")
                        
                        if diff < 50:  # 50%p 이내
                            print(f"    ★ 수기 계산과 매우 유사!")
                            
            except Exception as e:
                continue
        
        # 다른 시트나 다른 헤더 옵션도 확인
        print(f"\n다른 읽기 방법 시도:")
        for header_opt in [0, 2, 3]:
            try:
                df_alt = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=sheet_name, header=header_opt)
                
                # 1139.95에 가까운 값 찾기
                for col in df_alt.columns:
                    try:
                        data = pd.to_numeric(df_alt[col], errors='coerce').dropna()
                        for val in data:
                            if 1100 < val < 1200:  # 1139.95 근처
                                diff = abs(val - target_return)
                                if diff < 20:
                                    print(f"  헤더{header_opt}, {col}: {val:.2f}% (차이: {diff:.2f}%p) ★")
                    except:
                        continue
                        
            except:
                continue
                
    except Exception as e:
        print(f"추가 분석 오류: {e}")

if __name__ == "__main__":
    # 시뮬레이션 파일로 백테스팅 검증
    result = verify_with_simulation_file()
    
    # 정확한 계산 방식 탐색
    find_exact_calculation_method()