import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import seaborn as sns
import os
import sys

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# RSI 계절 분류 기준 (수정 가능한 변수)
RSI_THRESHOLDS = {
    'summer': 70,  # RSI >= 70: 여름 (과매수)
    'winter': 30   # RSI < 30: 겨울 (과매도)
    # 30 <= RSI < 50: 가을 (하락 추세)
    # 50 <= RSI < 70: 봄 (상승 추세)
}

def load_sp500_data(file_path):
    """S&P 500 스타일 지수 데이터를 로드합니다."""
    try:
        # Excel 파일 또는 CSV 파일 자동 감지
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # 첫 번째 컬럼을 날짜로 설정
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        
        # NaT 값 제거
        df = df[df.index.notna()]
        
        # 날짜 순으로 정렬 (오래된 것부터)
        df.sort_index(inplace=True)
        
        print(f"S&P 500 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        if len(df) > 0:
            print(f"기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"사용 가능한 스타일: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"S&P 500 데이터 로딩 오류: {e}")
        return None

def load_rsi_data(file_path):
    """RSI 데이터를 로드합니다."""
    try:
        df = pd.read_excel(file_path, skiprows=1)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
rsi_series = df['RSI'].dropna()
        print(f"RSI 데이터 로딩 완료: {len(rsi_series)}개 데이터 포인트")
        print(f"RSI 기간: {rsi_series.index[0].strftime('%Y-%m-%d')} ~ {rsi_series.index[-1].strftime('%Y-%m-%d')}")
        return rsi_series
    except Exception as e:
        print(f"RSI 데이터 로딩 오류: {e}")
        return None

def classify_market_season(rsi_value, thresholds=RSI_THRESHOLDS):
    """RSI 값을 기반으로 시장 계절을 분류합니다."""
    if pd.isna(rsi_value):
        return np.nan
    elif rsi_value >= thresholds['summer']:
        return '여름'
    elif rsi_value >= 50:
        return '봄'
    elif rsi_value >= thresholds['winter']:
        return '가을'
    else:
        return '겨울'

def calculate_buy_and_hold_returns(sp500_df, start_date, end_date):
    """개별 전략별 단순 보유 수익률을 계산합니다 (1998년 12월 말 기준)."""
    
    # 기준일 설정 (1998년 12월 말)
    base_date = pd.Timestamp('1998-12-31')
    
    # 실제 거래일 찾기
    try:
        # 기준일에 가장 가까운 거래일 찾기
        available_dates = sp500_df.index
        
        # 기준일 이후 첫 번째 날짜 찾기
        base_candidates = available_dates[available_dates >= base_date]
        if len(base_candidates) == 0:
            # 기준일보다 이전 날짜 중 가장 늦은 날짜
            base_actual = available_dates.max()
        else:
            base_actual = base_candidates.min()
            
        # 종료일 이전 마지막 날짜 찾기  
        end_candidates = available_dates[available_dates <= end_date]
        if len(end_candidates) == 0:
            # 종료일보다 이후 날짜 중 가장 빠른 날짜
            end_actual = available_dates.min()
        else:
            end_actual = end_candidates.max()
        
        print(f"\n=== 단순 보유 수익률 계산 ===")
        print(f"기준일: {base_actual.strftime('%Y-%m-%d')}")
        print(f"종료일: {end_actual.strftime('%Y-%m-%d')}")
        
    except (KeyError, IndexError) as e:
        print(f"날짜 설정 오류: {e}")
        return None
    
    results = {}
    
    for style in sp500_df.columns:
        try:
            # 초기값 설정
            if 'Momentum' in style or 'Quality' in style:
                # 모멘텀과 퀄리티는 초기값 100
                base_value = 100.0
            else:
                # 기타 전략은 1998년 12월 말 종가 사용
                base_value = sp500_df.loc[base_actual, style]
            
            # 종료일 값
            end_value = sp500_df.loc[end_actual, style]
            
            # 총수익률 계산
            total_return = (end_value / base_value) - 1
            
            results[style] = {
                'base_value': base_value,
                'end_value': end_value,
                'total_return': total_return,
                'base_date': base_actual,
                'end_date': end_actual
            }
            
            print(f"{style:<15}: {base_value:>8.2f} → {end_value:>8.2f} ({total_return:>8.2%})")
            
        except Exception as e:
            print(f"{style} 계산 오류: {e}")
            continue
    
    return results

def run_dynamic_strategy(sp500_df, rsi_series, start_date, end_date, 
                        initial_capital=10000000, strategy_rules=None, special_strategy=None):
    """동적 RSI 기반 로테이션 전략을 실행합니다."""
    
    if strategy_rules is None:
        print("오류: 전략 규칙이 제공되지 않았습니다.")
        return None
    
    print(f"\n=== 동적 전략 백테스팅 ===")
    print(f"기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"초기 자본: {initial_capital:,}원")
    print(f"전략 규칙: {strategy_rules}")
    if special_strategy:
        print(f"횡보 구간 특별 전략: {special_strategy}")
    
    # 월말 데이터만 필터링
    monthly_sp500 = sp500_df.resample('M').last()
    monthly_rsi = rsi_series.resample('M').last()
    
    # 백테스팅 기간으로 필터링
    monthly_sp500 = monthly_sp500[(monthly_sp500.index >= start_date) & 
                                  (monthly_sp500.index <= end_date)]
    monthly_rsi = monthly_rsi[(monthly_rsi.index >= start_date) & 
                              (monthly_rsi.index <= end_date)]
    
    print(f"월별 데이터 포인트: SP500 {len(monthly_sp500)}개, RSI {len(monthly_rsi)}개")
    
    # 결과 저장 변수
    portfolio_values = []
    transactions = []
    monthly_returns = []
    current_style = None
    current_shares = 0
    cash = initial_capital
    
    # 계절별 통계
    season_stats = {'여름': [], '봄': [], '가을': [], '겨울': []}
    
    # 횡보 구간 감지를 위한 변수
    season_history = []  # 최근 계절 기록
    in_sideways_mode = False  # 횡보 모드 여부
    
    for i, date in enumerate(monthly_sp500.index):
        # 현재 월의 RSI로 계절 판단
        if date in monthly_rsi.index:
            current_rsi = monthly_rsi.loc[date]
            current_season = classify_market_season(current_rsi)
        else:
            # RSI 데이터가 없으면 이전 월 RSI 사용하거나 기본값
            if i > 0 and len(monthly_returns) > 0:
                current_season = monthly_returns[-1]['season']
            else:
                current_season = '봄'  # 기본값
            current_rsi = 50.0
        
        # 계절 기록 업데이트 (횡보 구간 감지용)
        if pd.notna(current_season):
            season_history.append(current_season)
            # 최근 5개월 기록만 유지
            if len(season_history) > 5:
                season_history.pop(0)
        
        # 횡보 구간 감지: 봄과 가을만 반복하는 구간
        if special_strategy:
            # 횡보 모드 진입 조건: 봄↔가을 전환을 감지하면 횡보 모드 시작
            if not in_sideways_mode:
                if len(season_history) >= 2:
                    prev_season = season_history[-2]
                    # 가을 → 봄 전환 또는 봄 → 가을 전환을 감지
                    if ((prev_season == '가을' and current_season == '봄') or 
                        (prev_season == '봄' and current_season == '가을')):
                        in_sideways_mode = True
                        print(f"  [횡보 모드 진입] {date.strftime('%Y-%m')}: {prev_season}→{current_season} 전환, 횡보 구간 시작")
            
            # 횡보 모드 종료 조건: 여름 또는 겨울 출현
            elif in_sideways_mode and current_season in ['여름', '겨울']:
                in_sideways_mode = False
                print(f"  [횡보 모드 종료] {date.strftime('%Y-%m')}: {current_season} 출현, 일반 모드로 복귀")
        
        # 투자할 스타일 결정
        if in_sideways_mode and special_strategy and current_season in special_strategy:
            # 횡보 구간에서는 특별 전략 사용
            target_style = special_strategy[current_season]
        elif current_season in strategy_rules:
            # 일반 구간에서는 기본 전략 사용
            target_style = strategy_rules[current_season]
        else:
            target_style = 'Quality'  # 기본값
        
        # 스타일명 매핑 (데이터 컬럼명과 매칭)
        style_mapping = {
            'Momentum': 'S&P500 Momentum',
            'Quality': 'S&P500 Quality', 
            'Low Vol': 'S&P500 Low Volatiltiy Index',
            'Growth': 'S&P500 Growth',
            'Value': 'S&P500 Value',
            'Dividend': 'S&P500 Div Aristocrt TR Index',
            'S&P500': 'S&P500'
        }
        
        actual_style = style_mapping.get(target_style, target_style)
        
        # 해당 스타일이 데이터에 없으면 사용 가능한 첫 번째 스타일 사용
        if actual_style not in monthly_sp500.columns:
            actual_style = monthly_sp500.columns[0]
        
        # 포트폴리오 가치 계산 (리밸런싱 전)
        if current_style and current_shares > 0:
            portfolio_value = current_shares * monthly_sp500.loc[date, current_style]
        else:
            portfolio_value = cash
        
        # 월별 수익률 계산 (첫 달 제외)
        if i > 0:
            prev_value = portfolio_values[-1]
            monthly_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
            
            # 이전 달 계절 통계에 추가
            prev_season = monthly_returns[-1]['season']
            season_stats[prev_season].append(monthly_return)
        else:
            monthly_return = 0
        
        # 리밸런싱 (스타일 변경시)
        if actual_style != current_style:
            # 매도
            if current_style and current_shares > 0:
                sell_price = monthly_sp500.loc[date, current_style]
                cash = current_shares * sell_price
                transactions.append({
                    'date': date,
                    'action': 'SELL',
                    'style': current_style,
                    'price': sell_price,
                    'shares': current_shares,
                    'value': cash
                })
            
            # 매수
            buy_price = monthly_sp500.loc[date, actual_style]
            current_shares = cash / buy_price
            current_style = actual_style
            cash = 0
            
            transactions.append({
                'date': date,
                'action': 'BUY',
                'style': current_style,
                'price': buy_price,
                'shares': current_shares,
                'value': current_shares * buy_price
            })
            
            # 리밸런싱 후 포트폴리오 가치 재계산
            portfolio_value = current_shares * monthly_sp500.loc[date, current_style]
        
        # 결과 저장
        portfolio_values.append(portfolio_value)
        monthly_returns.append({
            'date': date,
            'rsi': current_rsi,
            'season': current_season,
            'style': actual_style,
            'portfolio_value': portfolio_value,
            'monthly_return': monthly_return
        })
    
    # 최종 결과 계산
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital) - 1
    
    # 연평균 수익률 (CAGR)
    years = (monthly_sp500.index[-1] - monthly_sp500.index[0]).days / 365.25
    cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # 최대 낙폭 (MDD)
    portfolio_series = pd.Series(portfolio_values, index=monthly_sp500.index)
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / rolling_max) - 1
    mdd = drawdown.min()
    
    # 변동성
    returns_series = portfolio_series.pct_change().dropna()
    volatility = returns_series.std() * np.sqrt(12)  # 월별 → 연간
    
    # 샤프 비율
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'investment_years': years
    }
    
    print(f"동적 전략 결과:")
    print(f"  총수익률: {total_return:.2%}")
    print(f"  연평균수익률(CAGR): {cagr:.2%}")
    print(f"  최대낙폭(MDD): {mdd:.2%}")
    print(f"  샤프비율: {sharpe_ratio:.2f}")
    
    return {
        'portfolio_series': portfolio_series,
        'monthly_returns': pd.DataFrame(monthly_returns),
        'transactions': pd.DataFrame(transactions),
        'season_stats': season_stats,
        'metrics': metrics
    }

# 기본 전략 관련 함수들 제거됨 - 이제 커스텀 전략과 단순 보유 전략만 비교

def get_available_styles(sp500_df):
    """사용 가능한 스타일 지수 목록을 반환합니다."""
    styles = sp500_df.columns.tolist()
    style_mapping = {}
    
    for i, style in enumerate(styles, 1):
        style_mapping[i] = style
    
    return style_mapping

def display_style_menu(style_mapping):
    """스타일 선택 메뉴를 표시합니다."""
    print("\n=== 사용 가능한 S&P 500 스타일 지수 ===")
    for num, style in style_mapping.items():
        style_short = style.replace('S&P500 ', '').replace(' Index', '').replace(' TR', '')
        print(f"{num}. {style_short}")
    print("=" * 45)

def get_custom_strategy(sp500_df):
    """사용자로부터 커스텀 전략을 입력받습니다."""
    print("\n=== 커스텀 로테이션 전략 설정 ===")
    print("각 계절별로 원하는 S&P 500 스타일을 선택하세요.")
    print("계절별 의미:")
    print(f"  [봄] RSI {RSI_THRESHOLDS['winter']}-{RSI_THRESHOLDS['summer']}: 상승 추세")
    print(f"  [여름] RSI {RSI_THRESHOLDS['summer']}+: 과매수 상태")
    print(f"  [가을] RSI {RSI_THRESHOLDS['winter']}-50: 하락 추세")  
    print(f"  [겨울] RSI <{RSI_THRESHOLDS['winter']}: 과매도 상태")
    
    style_mapping = get_available_styles(sp500_df)
    display_style_menu(style_mapping)
    
    custom_strategy = {}
    seasons = ['봄', '여름', '가을', '겨울']
    season_icons = {'봄': '[봄]', '여름': '[여름]', '가을': '[가을]', '겨울': '[겨울]'}
    
    try:
        for season in seasons:
            while True:
                try:
                    choice = int(input(f"\n{season_icons[season]} {season} 계절 스타일 선택 (번호 입력): "))
                    if choice in style_mapping:
                        # 실제 컬럼명을 약식 이름으로 매핑
                        actual_style = style_mapping[choice]
                        short_name = actual_style.replace('S&P500 ', '').replace(' Index', '').replace(' TR', '')
                        
                        if 'Momentum' in actual_style:
                            custom_strategy[season] = 'Momentum'
                        elif 'Quality' in actual_style:
                            custom_strategy[season] = 'Quality'
                        elif 'Low Vol' in actual_style:
                            custom_strategy[season] = 'Low Vol'
                        elif 'Growth' in actual_style:
                            custom_strategy[season] = 'Growth'
                        elif 'Value' in actual_style:
                            custom_strategy[season] = 'Value'
                        elif 'Div' in actual_style:
                            custom_strategy[season] = 'Dividend'
                        else:
                            custom_strategy[season] = 'Quality'  # 기본값
                            
                        print(f"  ✓ {season}: {short_name}")
                        break
                    else:
                        print("잘못된 번호입니다. 다시 선택해주세요.")
                except ValueError:
                    print("숫자를 입력해주세요.")
        
        # 특별 조건 설정 (횡보 구간)
        print("\n=== 특별 조건 설정 (고급) ===")
        print("봄-가을 횡보 구간에서 다른 전략을 사용하시겠습니까?")
        print("(봄↔가을 전환 후 여름/겨울 없이 봄-가을만 반복하는 구간에서 특별 전략 적용)")
        
        use_special = input("특별 조건 사용 (y/n, 기본값: n): ").lower().strip()
        special_strategy = None
        
        if use_special == 'y':
            print("\n횡보 구간 특별 전략 설정:")
            print("봄↔가을 전환 후 봄-가을 횡보 구간에서 아래 전략을 사용합니다.")
            
            special_strategy = {}
            
            # 횡보 구간에서의 봄 전략
            while True:
                try:
                    choice = int(input(f"\n횡보 구간 {season_icons['봄']} 봄 스타일 선택 (번호 입력): "))
                    if choice in style_mapping:
                        actual_style = style_mapping[choice]
                        if 'Momentum' in actual_style:
                            special_strategy['봄'] = 'Momentum'
                        elif 'Quality' in actual_style:
                            special_strategy['봄'] = 'Quality'
                        elif 'Low Vol' in actual_style:
                            special_strategy['봄'] = 'Low Vol'
                        elif 'Growth' in actual_style:
                            special_strategy['봄'] = 'Growth'
                        elif 'Value' in actual_style:
                            special_strategy['봄'] = 'Value'
                        elif 'Div' in actual_style:
                            special_strategy['봄'] = 'Dividend'
                        else:
                            special_strategy['봄'] = 'Quality'
                        
                        short_name = actual_style.replace('S&P500 ', '').replace(' Index', '').replace(' TR', '')
                        print(f"  ✓ 횡보 구간 봄: {short_name}")
                        break
                    else:
                        print("잘못된 번호입니다. 다시 선택해주세요.")
                except ValueError:
                    print("숫자를 입력해주세요.")
            
            # 횡보 구간에서의 가을 전략
            while True:
                try:
                    choice = int(input(f"\n횡보 구간 {season_icons['가을']} 가을 스타일 선택 (번호 입력): "))
                    if choice in style_mapping:
                        actual_style = style_mapping[choice]
                        if 'Momentum' in actual_style:
                            special_strategy['가을'] = 'Momentum'
                        elif 'Quality' in actual_style:
                            special_strategy['가을'] = 'Quality'
                        elif 'Low Vol' in actual_style:
                            special_strategy['가을'] = 'Low Vol'
                        elif 'Growth' in actual_style:
                            special_strategy['가을'] = 'Growth'
                        elif 'Value' in actual_style:
                            special_strategy['가을'] = 'Value'
                        elif 'Div' in actual_style:
                            special_strategy['가을'] = 'Dividend'
                        else:
                            special_strategy['가을'] = 'Quality'
                        
                        short_name = actual_style.replace('S&P500 ', '').replace(' Index', '').replace(' TR', '')
                        print(f"  ✓ 횡보 구간 가을: {short_name}")
                        break
                    else:
                        print("잘못된 번호입니다. 다시 선택해주세요.")
                except ValueError:
                    print("숫자를 입력해주세요.")
        
        # 전략명 입력
        strategy_name = input(f"\n전략명을 입력하세요 (기본값: '사용자 전략'): ").strip()
        if not strategy_name:
            strategy_name = '사용자 전략'
        
        # 확인 출력
        print(f"\n=== '{strategy_name}' 전략 구성 ===")
        print("기본 전략:")
        for season in seasons:
            print(f"  {season_icons[season]} {season}: {custom_strategy[season]}")
        
        if special_strategy:
            print("\n특별 조건 (봄-가을 횡보 구간):")
            print(f"  {season_icons['봄']} 횡보 봄: {special_strategy['봄']}")
            print(f"  {season_icons['가을']} 횡보 가을: {special_strategy['가을']}")
        
        confirm = input("\n이 설정으로 진행하시겠습니까? (y/n): ").lower()
        if confirm != 'y':
            print("전략 설정을 취소합니다.")
            return None
        
        return strategy_name, custom_strategy, special_strategy
        
    except KeyboardInterrupt:
        print("\n전략 설정을 취소합니다.")
        return None

def get_predefined_strategies():
    """기본 제공 전략들을 반환합니다 - 각 스타일별 단순 보유 전략."""
    # 단순 보유 전략: 한 가지 스타일만 계속 보유
    return {
        'Momentum 보유': {
            '여름': 'Momentum',
            '봄': 'Momentum',
            '가을': 'Momentum',
            '겨울': 'Momentum'
        },
        'Quality 보유': {
            '여름': 'Quality',
            '봄': 'Quality',
            '가을': 'Quality',
            '겨울': 'Quality'
        },
        'Growth 보유': {
            '여름': 'Growth',
            '봄': 'Growth',
            '가을': 'Growth',
            '겨울': 'Growth'
        },
        'Value 보유': {
            '여름': 'Value',
            '봄': 'Value',
            '가을': 'Value',
            '겨울': 'Value'
        },
        'Low Vol 보유': {
            '여름': 'Low Vol',
            '봄': 'Low Vol',
            '가을': 'Low Vol',
            '겨울': 'Low Vol'
        },
        'Dividend 보유': {
            '여름': 'Dividend',
            '봄': 'Dividend',
            '가을': 'Dividend',
            '겨울': 'Dividend'
        }
    }

def run_strategy_comparison(sp500_df, rsi_series, start_date, end_date, 
                          custom_strategy_name, custom_strategy_rules, initial_capital=10000000, 
                          custom_special_strategy=None):
    """커스텀 전략과 기본 전략들을 비교합니다."""
    all_strategies = get_predefined_strategies()
    all_strategies[custom_strategy_name] = custom_strategy_rules
    
    results = {}
    
    print(f"\n=== 전략 비교 백테스팅 실행 ===")
    print(f"커스텀 전략: {custom_strategy_name}")
    if custom_special_strategy:
        print("특별 조건 (봄-가을 횡보 구간) 포함")
    print("기본 전략들과 성과를 비교합니다...\n")
    
    for strategy_name, strategy_rules in all_strategies.items():
        print(f"{strategy_name} 실행 중...")
        
        # 커스텀 전략인 경우에만 특별 조건 적용
        special_strategy = custom_special_strategy if strategy_name == custom_strategy_name else None
        
        result = run_dynamic_strategy(
            sp500_df, rsi_series, start_date, end_date, initial_capital, 
            strategy_rules, special_strategy
        )
        
        results[strategy_name] = {
            'portfolio_series': result['portfolio_series'],
            'transactions': result['transactions'],
            'season_stats': result['season_stats'],
            'metrics': result['metrics'],
            'strategy_rules': strategy_rules,
            'special_strategy': special_strategy,
            'is_custom': strategy_name == custom_strategy_name
        }
        
        print(f"  최종 수익률: {result['metrics']['total_return']:.2%}")
        print(f"  CAGR: {result['metrics']['cagr']:.2%}")
        print(f"  샤프 비율: {result['metrics']['sharpe_ratio']:.2f}")
    
    return results

def create_strategy_comparison_chart(results, custom_strategy_name, buy_hold_results):
    """전략 비교 차트를 생성합니다."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    
    # 커스텀 전략 강조 색상
    colors = {}
    for i, strategy_name in enumerate(results.keys()):
        if strategy_name == custom_strategy_name:
            colors[strategy_name] = 'red'
        else:
            colors[strategy_name] = plt.cm.Set3(i / len(results))
    
    # 1. 포트폴리오 가치 변화
    ax1 = axes[0, 0]
    for strategy_name, result in results.items():
        linewidth = 3 if strategy_name == custom_strategy_name else 2
        alpha = 1.0 if strategy_name == custom_strategy_name else 0.7
        ax1.plot(result['portfolio_series'].index, result['portfolio_series'].values, 
                label=strategy_name, linewidth=linewidth, alpha=alpha, color=colors[strategy_name])
    
    ax1.set_title('전략별 포트폴리오 가치 변화', fontsize=14, fontweight='bold')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 2. 총수익률 비교
    ax2 = axes[0, 1]
    strategies = list(results.keys())
    total_returns = [results[s]['metrics']['total_return'] * 100 for s in strategies]
    bar_colors = [colors[s] for s in strategies]
    
    bars = ax2.bar(strategies, total_returns, color=bar_colors, alpha=0.8)
    ax2.set_title('전략별 총수익률', fontsize=14, fontweight='bold')
    ax2.set_ylabel('총수익률 (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        if strategy == custom_strategy_name:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # 3. 단순 보유 vs 동적 전략 비교
    ax3 = axes[1, 0]
    
    # 최고 성과 단순 보유 전략과 커스텀 전략 비교
    best_buy_hold = max(buy_hold_results.items(), key=lambda x: x[1]['total_return'])
    custom_result = results[custom_strategy_name]
    
    comparison_data = {
        f'최고 단순보유\n({best_buy_hold[0].replace("S&P500 ", "")})': best_buy_hold[1]['total_return'] * 100,
        f'커스텀 전략\n({custom_strategy_name})': custom_result['metrics']['total_return'] * 100
    }
    
    bars = ax3.bar(comparison_data.keys(), comparison_data.values(), 
                   color=['blue', 'red'], alpha=0.8)
    ax3.set_title('단순보유 vs 커스텀 전략', fontsize=14, fontweight='bold')
    ax3.set_ylabel('총수익률 (%)')
    
    for bar, (name, ret) in zip(bars, comparison_data.items()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{ret:.1f}%', ha='center', va='bottom')
    
    # 4. CAGR 비교  
    ax4 = axes[1, 1]
    cagr_values = [results[s]['metrics']['cagr'] * 100 for s in strategies]
    bars = ax4.bar(strategies, cagr_values, color=bar_colors, alpha=0.8)
    ax4.set_title('전략별 연평균 수익률 (CAGR)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('CAGR (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        if strategy == custom_strategy_name:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # 5. 샤프 비율 비교
    ax5 = axes[2, 0]
    sharpe_values = [results[s]['metrics']['sharpe_ratio'] for s in strategies]
    bars = ax5.bar(strategies, sharpe_values, color=bar_colors, alpha=0.8)
    ax5.set_title('전략별 샤프 비율', fontsize=14, fontweight='bold')
    ax5.set_ylabel('샤프 비율')
    ax5.tick_params(axis='x', rotation=45)
    
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
        if strategy == custom_strategy_name:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # 6. 커스텀 전략 구성
    ax6 = axes[2, 1]
    custom_rules = results[custom_strategy_name]['strategy_rules']
    seasons_list = ['봄', '여름', '가을', '겨울']
    season_icons = {'봄': '[봄]', '여름': '[여름]', '가을': '[가을]', '겨울': '[겨울]'}
    
    ax6.axis('off')
    ax6.set_title(f'{custom_strategy_name} 구성', fontsize=14, fontweight='bold')
    
    y_pos = 0.8
    for season in seasons_list:
        style = custom_rules[season]
        ax6.text(0.1, y_pos, f"{season_icons[season]} {season}: {style}", 
                fontsize=12, transform=ax6.transAxes)
        y_pos -= 0.15
    
    # 특별 전략이 있다면 표시
    if 'special_strategy' in results[custom_strategy_name] and results[custom_strategy_name]['special_strategy']:
        special = results[custom_strategy_name]['special_strategy']
        ax6.text(0.1, y_pos, "\n횡보 구간:", fontsize=10, transform=ax6.transAxes, fontweight='bold')
        y_pos -= 0.1
        for season, style in special.items():
            ax6.text(0.1, y_pos, f"  {season_icons[season]} {season}: {style}", 
                    fontsize=10, transform=ax6.transAxes, color='red')
            y_pos -= 0.08
    
    # 7. 성과 요약
    ax7 = axes[3, 0]
    ax7.axis('off')
    ax7.set_title(f'{custom_strategy_name} 성과 요약', fontsize=14, fontweight='bold')
    
    custom_metrics = results[custom_strategy_name]['metrics']
    summary_text = [
        f"투자기간: {custom_metrics['investment_years']:.1f}년",
        f"총 수익률: {custom_metrics['total_return']:.1%}",
        f"연평균 수익률: {custom_metrics['cagr']:.2%}",
        f"최대 낙폭: {custom_metrics['mdd']:.1%}",
        f"변동성: {custom_metrics['volatility']:.1%}",
        f"샤프 비율: {custom_metrics['sharpe_ratio']:.2f}",
        "",
        f"거래 횟수: {len(results[custom_strategy_name]['transactions'])}회"
    ]
    
    y_pos = 0.9
    for text in summary_text:
        ax7.text(0.1, y_pos, text, fontsize=12, transform=ax7.transAxes,
                fontweight='bold' if '총 수익률' in text or '연평균' in text else 'normal')
        y_pos -= 0.1
    
    # 8. 수익률 비교 요약 차트
    ax8 = axes[3, 1]
    
    # 커스텀 전략과 최고 성과 전략 비교
    best_strategy = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
    if best_strategy[0] != custom_strategy_name:
        comparison_strategies = [custom_strategy_name, best_strategy[0]]
    else:
        # 커스텀이 최고면 2등과 비교
        sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['total_return'], reverse=True)
        comparison_strategies = [custom_strategy_name, sorted_results[1][0]]
    
    total_rets = [results[s]['metrics']['total_return'] * 100 for s in comparison_strategies]
    cagr_vals = [results[s]['metrics']['cagr'] * 100 for s in comparison_strategies]
    
    x = range(len(comparison_strategies))
    width = 0.35
    
    bars1 = ax8.bar([i - width/2 for i in x], total_rets, width, label='총수익률 (%)', alpha=0.8, color='skyblue')
    bars2 = ax8.bar([i + width/2 for i in x], cagr_vals, width, label='CAGR (%)', alpha=0.8, color='lightcoral')
    
    ax8.set_title('커스텀 전략 vs 최고 성과 전략', fontsize=14, fontweight='bold')
    ax8.set_ylabel('수익률 (%)')
    ax8.set_xticks(x)
    ax8.set_xticklabels([s.replace(' 보유', '') for s in comparison_strategies], rotation=45)
    ax8.legend()
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'strategy_comparison_{custom_strategy_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n전략 비교 차트가 저장되었습니다: strategy_comparison_{custom_strategy_name.replace(' ', '_')}.png")

def main():
    """메인 함수 - 고도화된 백테스팅을 실행합니다."""
    
    print("="*60)
    print("     고도화된 투자 백테스팅 시스템 v1.0")
    print("="*60)
    
    try:
        # 1. 데이터 로딩
        print("\n1. 데이터 로딩 중...")
        
        # S&P 500 데이터 로딩
        sp500_file = resource_path('sp500_data.xlsx')
        if not os.path.exists(sp500_file):
            print(f"오류: sp500_data.xlsx 파일을 찾을 수 없습니다. 경로: {sp500_file}")
            return
        
        sp500_df = load_sp500_data(sp500_file)
        if sp500_df is None:
            print("오류: S&P 500 데이터 로딩 실패")
            return
        
        # RSI 데이터 로딩
        rsi_file = resource_path('RSI_DATE.xlsx')
        if not os.path.exists(rsi_file):
            print(f"오류: RSI_DATE.xlsx 데이터 파일을 찾을 수 없습니다. 경로: {rsi_file}")
            return
        
        rsi_series = load_rsi_data(rsi_file)
        if rsi_series is None:
            return
        
        # 2. 백테스팅 기간 설정
        start_date = pd.Timestamp('1999-01-01')
        end_date = pd.Timestamp('2025-06-30')
        
        print(f"\n2. 백테스팅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 3. 단순 보유 수익률 계산
        print(f"\n3. 개별 전략 단순 보유 수익률 계산 중...")
        buy_hold_results = calculate_buy_and_hold_returns(sp500_df, start_date, end_date)
        
        if buy_hold_results is None:
            print("단순 보유 수익률 계산 실패")
            return
        
        # 커스텀 전략 생성 및 비교만 지원
        print(f"\n4. 커스텀 전략 생성")
        custom_result = get_custom_strategy(sp500_df)
        if custom_result is None:
            print("커스텀 전략 생성이 취소되었습니다.")
            return
        
        # 반환값 처리 (특별 전략 포함 여부에 따라)
        if len(custom_result) == 3:
            custom_strategy_name, custom_strategy_rules, special_strategy = custom_result
        else:
            custom_strategy_name, custom_strategy_rules = custom_result
            special_strategy = None
        
        print(f"\n5. 전략 비교 백테스팅 실행 중...")
        results = run_strategy_comparison(
            sp500_df, rsi_series, start_date, end_date,
            custom_strategy_name, custom_strategy_rules, 10000000, special_strategy
        )
        
        # 커스텀 전략 상세 결과 출력
        custom_metrics = results[custom_strategy_name]['metrics']
        print(f"\n=== '{custom_strategy_name}' 상세 결과 ===")
        print(f"총수익률: {custom_metrics['total_return']:.2%}")
        print(f"연평균수익률(CAGR): {custom_metrics['cagr']:.2%}")
        print(f"최대낙폭(MDD): {custom_metrics['mdd']:.2%}")
        print(f"샤프비율: {custom_metrics['sharpe_ratio']:.2f}")
        
        # 순위 분석 - CAGR 기준
        print(f"\n=== 전략 순위 비교 (CAGR 기준) ===")
        cagr_ranking = sorted(results.items(), key=lambda x: x[1]['metrics']['cagr'], reverse=True)
        for rank, (strategy, result) in enumerate(cagr_ranking, 1):
            marker = " ★" if strategy == custom_strategy_name else ""
            total_ret = result['metrics']['total_return']
            cagr = result['metrics']['cagr']
            print(f"{rank}위: {strategy} (총수익률: {total_ret:.1%}, CAGR: {cagr:.2%}){marker}")
        
        # 순위 분석 - 총수익률 기준
        print(f"\n=== 전략 순위 비교 (총수익률 기준) ===")
        total_ranking = sorted(results.items(), key=lambda x: x[1]['metrics']['total_return'], reverse=True)
        for rank, (strategy, result) in enumerate(total_ranking, 1):
            marker = " ★" if strategy == custom_strategy_name else ""
            total_ret = result['metrics']['total_return']
            cagr = result['metrics']['cagr']
            print(f"{rank}위: {strategy} (총수익률: {total_ret:.1%}, CAGR: {cagr:.2%}){marker}")
        
        # 시각화
        print(f"\n6. 전략 비교 차트 생성 중...")
        create_strategy_comparison_chart(results, custom_strategy_name, buy_hold_results)
        
        print(f"\n{'='*60}")
        print("     백테스팅 완료!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n프로그램 실행 중 오류가 발생했습니다: {e}")
    finally:
        input("\n결과를 확인하고 Enter 키를 누르면 프로그램이 종료됩니다...")


if __name__ == "__main__":
    main()
