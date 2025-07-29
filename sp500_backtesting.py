"""
S&P 500 스타일 지수 백테스팅 프로그램

사용자가 지정한 기간과 S&P 500 스타일 지수를 선택하여 
월별 투자 시뮬레이션(백테스팅)을 실행하고, 
그 결과를 수치와 그래프로 보여주는 프로그램입니다.
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
    print("\n=== S&P 500 스타일 지수 백테스팅 프로그램 ===")
    
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
        
        # 입력값 검증
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 28)  # 월말 날짜는 나중에 조정
        
        if start_date >= end_date:
            raise ValueError("시작 날짜가 종료 날짜보다 늦습니다.")
        
        if index_choice < 0 or index_choice >= len(df.columns):
            raise ValueError("잘못된 지수 번호입니다.")
        
        if initial_capital <= 0:
            raise ValueError("초기 투자 원금은 0보다 커야 합니다.")
        
        return {
            'start_year': start_year,
            'start_month': start_month,
            'end_year': end_year,
            'end_month': end_month,
            'selected_index': selected_index,
            'initial_capital': initial_capital
        }
    
    except (ValueError, IndexError) as e:
        print(f"입력 오류: {e}")
        return None
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
        return None

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

def run_backtesting(df, selected_index, initial_capital):
    """
    백테스팅을 실행합니다.
    
    Args:
        df (pd.DataFrame): 필터링된 데이터프레임
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
        
    Returns:
        pd.Series: 월별 포트폴리오 가치 시계열
    """
    # 선택된 지수의 월별 수익률 계산
    index_data = df[selected_index].dropna()
    monthly_returns = index_data.pct_change().dropna()
    
    # 포트폴리오 가치 추적
    portfolio_values = [initial_capital]
    current_value = initial_capital
    
    # 월별 백테스팅 실행
    for return_rate in monthly_returns:
        current_value = current_value * (1 + return_rate)
        portfolio_values.append(current_value)
    
    # 날짜와 함께 시리즈로 변환
    dates = [index_data.index[0]] + list(monthly_returns.index)
    portfolio_series = pd.Series(portfolio_values, index=dates)
    
    return portfolio_series

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

def print_results(metrics, selected_index, initial_capital):
    """
    백테스팅 결과를 출력합니다.
    
    Args:
        metrics (dict): 성과 지표
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
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
    print("="*50)

def create_and_save_chart(portfolio_series, selected_index, initial_capital):
    """
    포트폴리오 가치 변화 차트를 생성하고 저장합니다.
    
    Args:
        portfolio_series (pd.Series): 포트폴리오 가치 시계열
        selected_index (str): 선택된 지수명
        initial_capital (float): 초기 투자 원금
    """
    plt.figure(figsize=(12, 8))
    
    # 그래프 그리기
    plt.plot(portfolio_series.index, portfolio_series.values, linewidth=2, color='blue')
    
    # 초기 투자 원금 선 추가
    plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label=f'초기 투자 원금: {initial_capital:,.0f}원')
    
    # 그래프 설정
    plt.title(f'{selected_index} 백테스팅 결과\n포트폴리오 가치 변화', fontsize=16, fontweight='bold')
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('포트폴리오 가치 (원)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Y축 포맷 설정 (콤마 구분)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # X축 날짜 포맷 설정
    plt.xticks(rotation=45)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 파일로 저장
    filename = 'backtesting_result.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n차트가 '{filename}' 파일로 저장되었습니다.")
    
    # 차트 표시 (터미널에서는 표시하지 않음)
    # plt.show()

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
        portfolio_series = run_backtesting(
            filtered_df, 
            user_input['selected_index'], 
            user_input['initial_capital']
        )
        
        # 5. 성과 지표 계산
        metrics = calculate_performance_metrics(portfolio_series, user_input['initial_capital'])
        
        # 6. 결과 출력
        print_results(metrics, user_input['selected_index'], user_input['initial_capital'])
        
        # 7. 차트 생성 및 저장
        create_and_save_chart(portfolio_series, user_input['selected_index'], user_input['initial_capital'])
        
        print("\n백테스팅이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"백테스팅 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()