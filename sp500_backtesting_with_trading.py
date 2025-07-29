"""
S&P 500 스타일 지수 백테스팅 프로그램 (월별 매수/매도 기능 포함)

사용자가 지정한 기간과 S&P 500 스타일 지수를 선택하여 
월별 투자 시뮬레이션(백테스팅)을 실행하고,
원하는 월에 매수/매도할 수 있는 기능을 제공합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """
    Excel 파일에서 S&P 500 지수 데이터를 로드하고 전처리합니다.
    
    Args:
        file_path (str): Excel 파일 경로
        
    Returns:
        pd.DataFrame: 날짜가 인덱스로 설정된 데이터프레임
    """
    try:
        # Excel 파일 로드
        df = pd.read_excel(file_path)
        
        # 첫 번째 컬럼을 날짜로 설정 (한글로 된 컬럼명 처리)
        date_column = df.columns[0]
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        
        # 날짜순으로 정렬 (오래된 순서대로)
        df.sort_index(inplace=True)
        
        print(f"데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        print(f"데이터 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
        
        return df
    
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return None

def display_available_indices(df):
    """
    사용 가능한 지수 목록을 출력합니다.
    
    Args:
        df (pd.DataFrame): 지수 데이터가 포함된 데이터프레임
    """
    print("\n=== 사용 가능한 S&P 500 스타일 지수 ===")
    for i, index_name in enumerate(df.columns, 1):
        print(f"{i}. {index_name}")
    print("=" * 40)

def get_user_input(df):
    """
    사용자로부터 백테스팅 파라미터를 입력받습니다.
    
    Args:
        df (pd.DataFrame): 지수 데이터가 포함된 데이터프레임
        
    Returns:
        dict: 사용자 입력 파라미터
    """
    print("\n=== S&P 500 스타일 지수 백테스팅 프로그램 (매수/매도 기능) ===")
    
    # 사용 가능한 지수 표시
    display_available_indices(df)
    
    try:
        # 시작 날짜 입력
        start_year = int(input("\n백테스팅 시작 연도 (YYYY): "))
        start_month = int(input("백테스팅 시작 월 (1-12): "))
        
        # 종료 날짜 입력
        end_year = int(input("백테스팅 종료 연도 (YYYY): "))
        end_month = int(input("백테스팅 종료 월 (1-12): "))
        
        # 지수 선택
        print("\n지수 선택:")
        index_choice = int(input("지수 번호를 선택하세요: ")) - 1
        selected_index = df.columns[index_choice]
        
        # 초기 투자 원금 입력
        initial_capital = float(input("초기 투자 원금 (원): "))
        
        # 거래 전략 선택
        print("\n거래 전략 선택:")
        print("1. 매월 정액 투자 (기존 방식)")
        print("2. 월별 매수/매도 지정")
        strategy_choice = int(input("전략을 선택하세요 (1 또는 2): "))
        
        trading_actions = None
        if strategy_choice == 2:
            trading_actions = get_trading_actions(start_year, start_month, end_year, end_month)
        
        # 입력값 검증
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 28)
        
        if start_date >= end_date:
            raise ValueError("시작 날짜가 종료 날짜보다 늦습니다.")
        
        if index_choice < 0 or index_choice >= len(df.columns):
            raise ValueError("잘못된 지수 번호입니다.")
        
        if initial_capital <= 0:
            raise ValueError("초기 투자 원금은 0보다 커야 합니다.")
        
        if strategy_choice not in [1, 2]:
            raise ValueError("잘못된 전략 선택입니다.")
        
        return {
            'start_year': start_year,
            'start_month': start_month,
            'end_year': end_year,
            'end_month': end_month,
            'selected_index': selected_index,
            'initial_capital': initial_capital,
            'strategy': strategy_choice,
            'trading_actions': trading_actions
        }
    
    except (ValueError, IndexError) as e:
        print(f"입력 오류: {e}")
        return None
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
        return None

def get_trading_actions(start_year, start_month, end_year, end_month):
    """
    월별 매수/매도 액션을 입력받습니다.
    
    Args:
        start_year, start_month, end_year, end_month: 백테스팅 기간
        
    Returns:
        dict: {날짜: 거래액션} 형태의 딕셔너리
    """
    print("\n=== 월별 거래 설정 ===")
    print("매월 거래 방식을 설정하세요:")
    print("- 매수: 양수 금액 입력 (예: 1000000)")
    print("- 매도: 음수 금액 입력 (예: -500000)")
    print("- 보유: 0 입력")
    print("- 종료: 'q' 입력")
    
    trading_actions = {}
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m')
        
        try:
            action_input = input(f"\n{date_str} 거래액 (매수: +, 매도: -, 보유: 0, 종료: q): ").strip()
            
            if action_input.lower() == 'q':
                break
            
            action_amount = float(action_input)
            trading_actions[date_str] = action_amount
            
        except ValueError:
            print("올바른 숫자를 입력하세요.")
            continue
        
        # 다음 달로 이동
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return trading_actions

def filter_data_by_period(df, start_year, start_month, end_year, end_month):
    """
    지정된 기간으로 데이터를 필터링합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        start_year, start_month, end_year, end_month: 기간 설정
        
    Returns:
        pd.DataFrame: 필터링된 데이터프레임
    """
    # 시작일과 종료일 설정 (datetime 객체로 생성)
    start_date = pd.Timestamp(start_year, start_month, 1)
    end_date = pd.Timestamp(end_year, end_month, 28)
    
    # 데이터 필터링 (boolean indexing 사용)
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df.loc[mask].copy()
    
    if len(filtered_df) == 0:
        raise ValueError("지정된 기간에 해당하는 데이터가 없습니다.")
    
    print(f"\n필터링된 데이터 기간: {filtered_df.index.min().strftime('%Y-%m-%d')} ~ {filtered_df.index.max().strftime('%Y-%m-%d')}")
    print(f"총 {len(filtered_df)}개 데이터 포인트")
    
    return filtered_df

def run_backtesting_basic(df, selected_index, initial_capital):
    """
    기존 방식의 백테스팅을 실행합니다. (매월 보유)
    
    Args:
        df (pd.DataFrame): 필터링된 데이터프레임
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
        
    Returns:
        tuple: (포트폴리오 가치 시계열, 거래 내역)
    """
    # 선택된 지수의 월별 수익률 계산
    index_data = df[selected_index].dropna()
    monthly_returns = index_data.pct_change().dropna()
    
    # 포트폴리오 가치 추적
    portfolio_values = [initial_capital]
    current_value = initial_capital
    
    # 거래 내역 추적
    transactions = [{'date': index_data.index[0], 'action': 'initial', 'amount': initial_capital, 'price': index_data.iloc[0], 'shares': initial_capital / index_data.iloc[0]}]
    
    # 월별 백테스팅 실행
    for i, return_rate in enumerate(monthly_returns):
        current_value = current_value * (1 + return_rate)
        portfolio_values.append(current_value)
    
    # 날짜와 함께 시리즈로 변환
    dates = [index_data.index[0]] + list(monthly_returns.index)
    portfolio_series = pd.Series(portfolio_values, index=dates)
    
    return portfolio_series, transactions

def run_backtesting_with_trading(df, selected_index, initial_capital, trading_actions):
    """
    월별 매수/매도 기능이 포함된 백테스팅을 실행합니다.
    
    Args:
        df (pd.DataFrame): 필터링된 데이터프레임
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
        trading_actions (dict): 월별 거래 액션
        
    Returns:
        tuple: (포트폴리오 가치 시계열, 거래 내역)
    """
    # 선택된 지수 데이터
    index_data = df[selected_index].dropna()
    
    # 포트폴리오 상태 초기화
    cash = initial_capital  # 현금
    shares = 0  # 보유 주식 수
    portfolio_values = []
    transactions = []
    
    # 각 월별로 백테스팅 실행
    for date in index_data.index:
        current_price = index_data[date]
        date_str = date.strftime('%Y-%m')
        
        # 해당 월의 거래 액션 확인
        if date_str in trading_actions:
            action_amount = trading_actions[date_str]
            
            if action_amount > 0:  # 매수
                if cash >= action_amount:
                    new_shares = action_amount / current_price
                    shares += new_shares
                    cash -= action_amount
                    transactions.append({
                        'date': date,
                        'action': 'buy',
                        'amount': action_amount,
                        'price': current_price,
                        'shares': new_shares,
                        'total_shares': shares,
                        'cash': cash
                    })
                else:
                    print(f"경고: {date_str}에 매수 자금 부족 (필요: {action_amount:,.0f}, 보유: {cash:,.0f})")
            
            elif action_amount < 0:  # 매도
                sell_amount = abs(action_amount)
                sell_shares = sell_amount / current_price
                
                if shares >= sell_shares:
                    shares -= sell_shares
                    cash += sell_amount
                    transactions.append({
                        'date': date,
                        'action': 'sell',
                        'amount': sell_amount,
                        'price': current_price,
                        'shares': sell_shares,
                        'total_shares': shares,
                        'cash': cash
                    })
                else:
                    # 보유 주식이 부족한 경우 전량 매도
                    if shares > 0:
                        sell_amount_actual = shares * current_price
                        cash += sell_amount_actual
                        transactions.append({
                            'date': date,
                            'action': 'sell',
                            'amount': sell_amount_actual,
                            'price': current_price,
                            'shares': shares,
                            'total_shares': 0,
                            'cash': cash
                        })
                        shares = 0
                        print(f"경고: {date_str}에 보유 주식 부족으로 전량 매도 (매도액: {sell_amount_actual:,.0f})")
                    else:
                        print(f"경고: {date_str}에 매도할 주식이 없습니다.")
        
        # 현재 포트폴리오 가치 계산 (현금 + 주식 가치)
        portfolio_value = cash + (shares * current_price)
        portfolio_values.append(portfolio_value)
    
    # 결과를 시리즈로 변환
    portfolio_series = pd.Series(portfolio_values, index=index_data.index)
    
    return portfolio_series, transactions

def calculate_performance_metrics(portfolio_series, initial_capital):
    """
    성과 지표를 계산합니다.
    
    Args:
        portfolio_series (pd.Series): 포트폴리오 가치 시계열
        initial_capital (float): 초기 투자 원금
        
    Returns:
        dict: 성과 지표들
    """
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    # 투자 기간 계산 (연 단위)
    start_date = portfolio_series.index[0]
    end_date = portfolio_series.index[-1]
    investment_years = (end_date - start_date).days / 365.25
    
    # CAGR 계산
    if investment_years > 0:
        cagr = (final_value / initial_capital) ** (1 / investment_years) - 1
    else:
        cagr = 0
    
    # MDD 계산
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max) - 1
    mdd = drawdown.min()
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'investment_years': investment_years
    }

def print_transactions(transactions):
    """
    거래 내역을 출력합니다.
    
    Args:
        transactions (list): 거래 내역 리스트
    """
    if not transactions:
        return
    
    print("\n" + "="*80)
    print("                    거래 내역")
    print("="*80)
    
    for i, trans in enumerate(transactions):
        date_str = trans['date'].strftime('%Y-%m-%d')
        action_str = {'buy': '매수', 'sell': '매도', 'initial': '초기투자'}.get(trans['action'], trans['action'])
        
        if trans['action'] == 'initial':
            print(f"{i+1:2d}. {date_str} | {action_str:6s} | 금액: {trans['amount']:>12,.0f}원")
        else:
            print(f"{i+1:2d}. {date_str} | {action_str:6s} | 금액: {trans['amount']:>12,.0f}원 | "
                  f"주가: {trans['price']:>8,.0f}원 | 주식: {trans['shares']:>8,.2f}주 | "
                  f"현금: {trans['cash']:>12,.0f}원")
    
    print("="*80)

def print_results(metrics, selected_index, initial_capital, transactions=None):
    """
    백테스팅 결과를 출력합니다.
    
    Args:
        metrics (dict): 성과 지표
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
        transactions (list): 거래 내역
    """
    print("\n" + "="*50)
    print("          백테스팅 결과")
    print("="*50)
    print(f"선택된 지수: {selected_index}")
    print(f"초기 투자 원금: {initial_capital:,.0f} 원")
    print(f"최종 자산: {metrics['final_value']:,.0f} 원")
    print(f"총 수익률: {metrics['total_return']:.2%}")
    print(f"연평균 복리 수익률 (CAGR): {metrics['cagr']:.2%}")
    print(f"최대 낙폭 (MDD): {metrics['mdd']:.2%}")
    print(f"투자 기간: {metrics['investment_years']:.1f} 년")
    if transactions:
        print(f"총 거래 횟수: {len([t for t in transactions if t['action'] != 'initial'])} 회")
    print("="*50)
    
    # 거래 내역 출력
    if transactions:
        print_transactions(transactions)

def create_and_save_chart(portfolio_series, selected_index, initial_capital, transactions=None):
    """
    포트폴리오 가치 변화 차트를 생성하고 저장합니다.
    
    Args:
        portfolio_series (pd.Series): 포트폴리오 가치 시계열
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
        transactions (list): 거래 내역
    """
    plt.figure(figsize=(14, 10))
    
    # 메인 차트
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_series.index, portfolio_series.values, linewidth=2, color='blue', label='포트폴리오 가치')
    
    # 초기 투자 원금 선 추가
    plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label=f'초기 투자 원금: {initial_capital:,.0f}원')
    
    # 거래 시점 표시
    if transactions:
        buy_dates = [t['date'] for t in transactions if t['action'] == 'buy']
        sell_dates = [t['date'] for t in transactions if t['action'] == 'sell']
        
        if buy_dates:
            buy_values = [portfolio_series[date] for date in buy_dates]
            plt.scatter(buy_dates, buy_values, color='green', marker='^', s=50, label='매수', zorder=5)
        
        if sell_dates:
            sell_values = [portfolio_series[date] for date in sell_dates]
            plt.scatter(sell_dates, sell_values, color='red', marker='v', s=50, label='매도', zorder=5)
    
    # 그래프 설정
    plt.title(f'{selected_index} 백테스팅 결과\n포트폴리오 가치 변화', fontsize=16, fontweight='bold')
    plt.ylabel('포트폴리오 가치 (원)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 거래량 차트 (있는 경우)
    if transactions and len([t for t in transactions if t['action'] != 'initial']) > 0:
        plt.subplot(2, 1, 2)
        
        trade_dates = []
        trade_amounts = []
        trade_colors = []
        
        for trans in transactions:
            if trans['action'] != 'initial':
                trade_dates.append(trans['date'])
                if trans['action'] == 'buy':
                    trade_amounts.append(trans['amount'])
                    trade_colors.append('green')
                else:  # sell
                    trade_amounts.append(-trans['amount'])
                    trade_colors.append('red')
        
        if trade_dates:
            plt.bar(trade_dates, trade_amounts, color=trade_colors, alpha=0.7, width=20)
            plt.title('월별 거래 내역', fontsize=14)
            plt.ylabel('거래 금액 (원)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # X축 날짜 포맷 설정
    plt.xticks(rotation=45)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 파일로 저장
    filename = 'backtesting_result_with_trading.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n차트가 '{filename}' 파일로 저장되었습니다.")

def main():
    """
    메인 함수 - 전체 백테스팅 프로세스를 실행합니다.
    """
    # 1. 데이터 로딩
    file_path = 'sp500_data.xlsx'
    df = load_data(file_path)
    
    if df is None:
        print("데이터 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 2. 사용자 입력 받기
    user_input = get_user_input(df)
    
    if user_input is None:
        print("입력이 올바르지 않습니다. 프로그램을 종료합니다.")
        return
    
    try:
        # 3. 데이터 필터링
        filtered_df = filter_data_by_period(
            df, 
            user_input['start_year'], 
            user_input['start_month'],
            user_input['end_year'], 
            user_input['end_month']
        )
        
        # 4. 백테스팅 실행
        print(f"\n백테스팅 실행 중... (지수: {user_input['selected_index']})")
        
        if user_input['strategy'] == 1:
            # 기존 방식 (매월 보유)
            portfolio_series, transactions = run_backtesting_basic(
                filtered_df, 
                user_input['selected_index'], 
                user_input['initial_capital']
            )
        else:
            # 월별 매수/매도 방식
            portfolio_series, transactions = run_backtesting_with_trading(
                filtered_df, 
                user_input['selected_index'], 
                user_input['initial_capital'],
                user_input['trading_actions']
            )
        
        # 5. 성과 지표 계산
        metrics = calculate_performance_metrics(portfolio_series, user_input['initial_capital'])
        
        # 6. 결과 출력
        print_results(metrics, user_input['selected_index'], user_input['initial_capital'], transactions)
        
        # 7. 차트 생성 및 저장
        create_and_save_chart(portfolio_series, user_input['selected_index'], user_input['initial_capital'], transactions)
        
        print("\n백테스팅이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"백테스팅 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()