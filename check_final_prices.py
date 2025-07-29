import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def check_final_style_prices():
    """각 스타일별 마지막 주가를 확인합니다."""
    print("=== 스타일별 마지막 주가 확인 ===")
    
    try:
        # S&P 500 데이터 로딩
        sp500_df = pd.read_excel('sp500_data.xlsx')
        date_column = sp500_df.columns[0]
        sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
        sp500_df.set_index(date_column, inplace=True)
        sp500_df.sort_index(inplace=True)
        
        print(f"S&P 500 데이터 기간: {sp500_df.index.min()} ~ {sp500_df.index.max()}")
        print(f"총 {len(sp500_df)}개 데이터 포인트")
        
        # 백테스팅 기간 설정
        start_date = pd.Timestamp(1999, 1, 1)
        end_date = pd.Timestamp(2025, 6, 30)
        
        # 해당 기간 데이터 필터링
        filtered_data = sp500_df.loc[start_date:end_date]
        print(f"\n백테스팅 기간 데이터: {len(filtered_data)}개")
        print(f"실제 기간: {filtered_data.index.min()} ~ {filtered_data.index.max()}")
        
        # 스타일별 마지막 가격 확인
        styles = {
            'S&P500 Growth': '성장',
            'S&P500 Value': '가치', 
            'S&P500 Momentum': '모멘텀',
            'S&P500 Quality': '퀄리티',
            'S&P500 Low Volatiltiy Index': '저변동성',
            'S&P500 Div Aristocrt TR Index': '배당귀족'
        }
        
        print(f"\n스타일별 마지막 주가 (백테스팅 종료 시점):")
        print(f"{'스타일':<30} {'마지막날짜':<12} {'마지막가격':<12} {'초기가격':<12} {'총수익률':<12}")
        print("-" * 85)
        
        for style_en, style_kr in styles.items():
            if style_en in filtered_data.columns:
                # 마지막 유효한 데이터 찾기
                style_data = filtered_data[style_en].dropna()
                
                if len(style_data) > 0:
                    last_date = style_data.index[-1]
                    last_price = style_data.iloc[-1]
                    first_price = style_data.iloc[0]
                    total_return = ((last_price / first_price) - 1) * 100
                    
                    print(f"{style_kr:<30} {last_date.strftime('%Y-%m-%d'):<12} {last_price:>10.2f} {first_price:>10.2f} {total_return:>10.1f}%")
                else:
                    print(f"{style_kr:<30} {'데이터없음':<12}")
            else:
                print(f"{style_kr:<30} {'컬럼없음':<12}")
        
        return filtered_data
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def check_simulation_final_prices():
    """시뮬레이션 파일의 마지막 가격도 확인합니다."""
    print(f"\n=== 시뮬레이션 파일 마지막 가격 확인 ===")
    
    try:
        # 시뮬레이션 파일 로딩
        sim_df = pd.read_excel('S&P 시뮬레이션.xlsx', sheet_name=0, header=1)
        date_col = sim_df.columns[0]
        sim_df[date_col] = pd.to_datetime(sim_df[date_col])
        sim_df.set_index(date_col, inplace=True)
        sim_df.sort_index(inplace=True)
        
        print(f"시뮬레이션 데이터 기간: {sim_df.index.min()} ~ {sim_df.index.max()}")
        
        # 스타일 관련 컬럼들
        sim_styles = {
            '퀄리티': 'Quality',
            '모멘텀': 'Momentum', 
            'S&P 로볼': 'Low Volatility'
        }
        
        print(f"\n시뮬레이션 파일 스타일별 마지막 가격:")
        print(f"{'스타일':<20} {'마지막날짜':<12} {'마지막가격':<12} {'초기가격':<12} {'총수익률':<12}")
        print("-" * 75)
        
        for col in sim_df.columns:
            col_str = str(col)
            for sim_style, eng_name in sim_styles.items():
                if sim_style in col_str:
                    style_data = pd.to_numeric(sim_df[col], errors='coerce').dropna()
                    
                    if len(style_data) > 0:
                        last_date = style_data.index[-1]
                        last_price = style_data.iloc[-1]
                        first_price = style_data.iloc[0]
                        total_return = ((last_price / first_price) - 1) * 100
                        
                        print(f"{sim_style:<20} {last_date.strftime('%Y-%m-%d'):<12} {last_price:>10.2f} {first_price:>10.2f} {total_return:>10.1f}%")
                    break
        
        return sim_df
        
    except Exception as e:
        print(f"시뮬레이션 파일 확인 오류: {e}")
        return None

def compare_data_sources():
    """두 데이터 소스의 가격을 비교합니다."""
    print(f"\n=== 데이터 소스별 가격 비교 ===")
    
    # 기존 데이터
    sp500_data = check_final_style_prices()
    
    # 시뮬레이션 데이터  
    sim_data = check_simulation_final_prices()
    
    if sp500_data is not None and sim_data is not None:
        print(f"\n=== 공통 날짜 가격 비교 (마지막 몇 개 날짜) ===")
        
        # 공통 날짜 찾기
        common_dates = sp500_data.index.intersection(sim_data.index)
        if len(common_dates) > 0:
            # 마지막 5개 날짜
            recent_dates = sorted(common_dates)[-5:]
            
            print(f"최근 5개 공통 날짜에서의 가격 비교:")
            print(f"{'날짜':<12} {'기존모멘텀':<12} {'시뮬모멘텀':<12} {'차이%':<8}")
            print("-" * 50)
            
            for date in recent_dates:
                try:
                    # 기존 데이터에서 모멘텀 가격
                    sp500_momentum = sp500_data.loc[date, 'S&P500 Momentum']
                    
                    # 시뮬레이션에서 모멘텀 가격 찾기
                    sim_momentum = None
                    for col in sim_data.columns:
                        if '모멘텀' in str(col):
                            sim_momentum = pd.to_numeric(sim_data.loc[date, col], errors='coerce')
                            break
                    
                    if sim_momentum is not None and pd.notna(sim_momentum):
                        diff_pct = ((sim_momentum - sp500_momentum) / sp500_momentum) * 100
                        print(f"{date.strftime('%Y-%m-%d'):<12} {sp500_momentum:>10.2f} {sim_momentum:>10.2f} {diff_pct:>6.2f}%")
                    
                except Exception as e:
                    print(f"{date.strftime('%Y-%m-%d'):<12} 비교 불가: {e}")

def check_portfolio_final_value():
    """현재 코드로 계산되는 포트폴리오의 최종 값을 확인합니다."""
    print(f"\n=== 포트폴리오 최종 값 상세 확인 ===")
    
    try:
        # 기존 백테스팅 코드 재실행해서 상세 정보 확인
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
        
        # 백테스팅 실행
        start_date = pd.Timestamp(1999, 1, 1)
        end_date = pd.Timestamp(2025, 6, 30)
        initial_capital = 10000000
        
        # 데이터 정렬 (기존 방식과 동일)
        sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
        rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
        
        sp500_filtered = sp500_df.loc[sp500_mask].copy()
        rsi_filtered = rsi_series.loc[rsi_mask].copy()
        
        # 월별 매칭
        sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
        rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
        rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
        
        common_periods = sorted(list(set(sp500_filtered['year_month']).intersection(set(rsi_filtered_df['year_month']))))
        
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
        
        # 전략 실행 (마지막 상태 확인)
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
        
        seasons = rsi_aligned.apply(classify_season)
        
        portfolio_value = initial_capital
        current_shares = 0
        current_style = None
        
        # 마지막 몇 개 거래 상세 추적
        total_trades = len(sp500_aligned)
        print(f"총 거래 기간: {total_trades}개월")
        
        for i, date in enumerate(sp500_aligned.index):
            season = seasons.loc[date] if date in seasons.index else np.nan
            
            if pd.isna(season):
                if current_style and current_shares > 0:
                    portfolio_value = current_shares * sp500_aligned.loc[date, current_style]
                continue
            
            target_style = strategy_rules[season]
            price = sp500_aligned.loc[date, target_style]
            
            if i == 0:
                current_style = target_style
                current_shares = portfolio_value / price
                portfolio_value = current_shares * price
            elif target_style != current_style:
                if current_style and current_shares > 0:
                    sell_price = sp500_aligned.loc[date, current_style]
                    cash = current_shares * sell_price
                    
                    current_style = target_style
                    current_shares = cash / price
                    portfolio_value = current_shares * price
            else:
                portfolio_value = current_shares * price
            
            # 마지막 10개 거래 출력
            if i >= total_trades - 10:
                rsi_val = rsi_aligned.loc[date]
                print(f"[{i+1:3d}] {date.strftime('%Y-%m')} RSI:{rsi_val:5.1f} {season} {target_style[:15]:15s} 가격:{price:8.2f} 포트폴리오:{portfolio_value:12,.0f}")
        
        print(f"\n최종 결과:")
        print(f"최종 포트폴리오 가치: {portfolio_value:,.0f}원")
        print(f"현재 보유 스타일: {current_style}")
        print(f"현재 보유 주식 수: {current_shares:.4f}주")
        print(f"마지막 주가: {price:.2f}")
        print(f"총 수익률: {((portfolio_value / initial_capital) - 1) * 100:.2f}%")
        
        return portfolio_value, current_style, current_shares, price
        
    except Exception as e:
        print(f"포트폴리오 확인 오류: {e}")
        return None

if __name__ == "__main__":
    # 모든 확인 실행
    compare_data_sources()
    final_result = check_portfolio_final_value()