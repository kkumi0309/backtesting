import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import seaborn as sns
import json
import os

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

def align_data(sp500_df, rsi_series, start_date, end_date):
    """S&P 500 데이터와 RSI 데이터를 월별로 매칭하여 정렬합니다."""
    # 기간 필터링
    sp500_mask = (sp500_df.index >= start_date) & (sp500_df.index <= end_date)
    rsi_mask = (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
    
    sp500_filtered = sp500_df.loc[sp500_mask].copy()
    rsi_filtered = rsi_series.loc[rsi_mask].copy()
    
    print(f"필터링 후 - SP500: {len(sp500_filtered)}개, RSI: {len(rsi_filtered)}개")
    
    # 월별 매칭을 위해 년-월 키 생성
    sp500_filtered['year_month'] = sp500_filtered.index.to_period('M')
    rsi_filtered_df = pd.DataFrame({'RSI': rsi_filtered})
    rsi_filtered_df['year_month'] = rsi_filtered.index.to_period('M')
    
    # S&P500 기준으로 모든 월 사용
    sp500_periods = sorted(list(set(sp500_filtered['year_month'])))
    
    if len(sp500_periods) == 0:
        raise ValueError("지정된 기간에 S&P 500 데이터가 없습니다.")
    
    print(f"처리할 월 수: {len(sp500_periods)}개")
    
    aligned_data = []
    aligned_rsi = []
    aligned_dates = []
    
    for i, period in enumerate(sp500_periods):
        sp500_month = sp500_filtered[sp500_filtered['year_month'] == period]
        rsi_month = rsi_filtered_df[rsi_filtered_df['year_month'] == period]
        
        if len(sp500_month) > 0:
            sp500_date = sp500_month.index[0]
            
            # RSI 값 찾기
            if len(rsi_month) > 0:
                rsi_value = rsi_month['RSI'].iloc[0]
            else:
                # RSI가 없는 경우 기본값 50 사용
                rsi_value = 50.0
                print(f"WARNING: {period}에 RSI 없음, 기본값 50 사용")
            
            aligned_dates.append(sp500_date)
            aligned_data.append(sp500_month.drop('year_month', axis=1).iloc[0])
            aligned_rsi.append(rsi_value)
    
    if not aligned_dates:
        raise ValueError("데이터 정렬 후 남은 데이터가 없습니다.")

    sp500_aligned = pd.DataFrame(aligned_data, index=aligned_dates)
    rsi_aligned = pd.Series(aligned_rsi, index=aligned_dates)
    
    print(f"데이터 정렬 완료: {len(aligned_dates)}개 데이터 포인트")
    return sp500_aligned, rsi_aligned

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
        print(f"{num}. {style}")
    print("=" * 45)

def get_custom_strategy():
    """사용자로부터 커스텀 전략을 입력받습니다."""
    print("\n=== 커스텀 로테이션 전략 설정 ===")
    print("각 계절별로 원하는 S&P 500 스타일을 선택하세요.")
    print("계절별 의미:")
    print("  [봄] 봄 (RSI 50-70): 상승 추세")
    print("  [여름] 여름 (RSI 70+): 과매수 상태")
    print("  [가을] 가을 (RSI 30-50): 하락 추세")  
    print("  [겨울] 겨울 (RSI <30): 과매도 상태")
    
    # 스타일 매핑 로드
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sp500_df = load_sp500_data(os.path.join(script_dir, 'sp500_data.xlsx'))
    if sp500_df is None:
        return None
    
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
                        custom_strategy[season] = style_mapping[choice]
                        print(f"  ✓ {season}: {style_mapping[choice]}")
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
        for season in seasons:
            print(f"{season_icons[season]} {season}: {custom_strategy[season]}")
        
        confirm = input("\n이 설정으로 진행하시겠습니까? (y/n): ").lower()
        if confirm != 'y':
            print("전략 설정을 취소합니다.")
            return None
        
        return strategy_name, custom_strategy
        
    except KeyboardInterrupt:
        print("\n전략 설정을 취소합니다.")
        return None

def save_custom_strategy(strategy_name, strategy_rules):
    """커스텀 전략을 파일로 저장합니다."""
    strategies_file = 'custom_strategies.json'
    
    # 기존 전략들 로드
    if os.path.exists(strategies_file):
        with open(strategies_file, 'r', encoding='utf-8') as f:
            saved_strategies = json.load(f)
    else:
        saved_strategies = {}
    
    # 새 전략 추가
    saved_strategies[strategy_name] = strategy_rules
    
    # 파일로 저장
    with open(strategies_file, 'w', encoding='utf-8') as f:
        json.dump(saved_strategies, f, ensure_ascii=False, indent=2)
    
    print(f"전략 '{strategy_name}'이 저장되었습니다.")

def load_saved_strategies():
    """저장된 커스텀 전략들을 로드합니다."""
    strategies_file = 'custom_strategies.json'
    
    if not os.path.exists(strategies_file):
        return {}
    
    try:
        with open(strategies_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def get_predefined_strategies():
    """기본 제공 전략들을 반환합니다."""
    return {
        '모멘텀 전략': {
            '여름': 'S&P500 Momentum',
            '봄': 'S&P500 Growth',
            '가을': 'S&P500 Quality',
            '겨울': 'S&P500 Value'
        },
        '안정성 전략': {
            '여름': 'S&P500 Low Volatiltiy Index',
            '봄': 'S&P500 Quality',
            '가을': 'S&P500 Div Aristocrt TR Index',
            '겨울': 'S&P500 Value'
        },
        '성장 중심 전략': {
            '여름': 'S&P500 Growth',
            '봄': 'S&P500 Momentum',
            '가을': 'S&P500 Growth',
            '겨울': 'S&P500 Quality'
        },
        '가치 중심 전략': {
            '여름': 'S&P500 Value',
            '봄': 'S&P500 Value',
            '가을': 'S&P500 Value',
            '겨울': 'S&P500 Div Aristocrt TR Index'
        },
        '배당 중심 전략': {
            '여름': 'S&P500 Div Aristocrt TR Index',
            '봄': 'S&P500 Quality',
            '가을': 'S&P500 Div Aristocrt TR Index',
            '겨울': 'S&P500 Value'
        }
    }

def run_rotation_strategy(df, strategy_name, strategy_rules, seasons, initial_capital=10000000):
    """로테이션 전략을 실행합니다."""
    portfolio_values = []
    transactions = []
    current_style = None
    current_shares = 0
    cash = 0
    
    season_stats = {'여름': [], '봄': [], '가을': [], '겨울': []}
    
    # 모든 계절이 동일한 스타일인지 확인 (Buy & Hold 최적화)
    unique_styles = set(strategy_rules.values())
    is_buy_and_hold = len(unique_styles) == 1
    
    if is_buy_and_hold:
        # Buy & Hold 전략: 처음에만 매수하고 끝
        single_style = list(unique_styles)[0]
        
        for i, date in enumerate(df.index):
            if i == 0:
                # 초기 투자
                current_style = single_style
                current_price = df.loc[date, current_style]
                current_shares = initial_capital / current_price
                cash = 0
                portfolio_value = initial_capital
                
                season = seasons.loc[date] if date in seasons.index else np.nan
                transactions.append({
                    'date': date,
                    'season': season,
                    'action': '초기투자',
                    'from_style': None,
                    'to_style': current_style,
                    'shares': current_shares,
                    'price': current_price,
                    'value': initial_capital
                })
            else:
                # 보유만 함
                current_price = df.loc[date, current_style]
                portfolio_value = current_shares * current_price
            
            # 수익률 계산 (계절별 통계용)
            if i > 0:
                prev_value = portfolio_values[-1]
                period_return = (portfolio_value / prev_value) - 1
                season = seasons.loc[date] if date in seasons.index else np.nan
                if not pd.isna(season):
                    season_stats[season].append(period_return)
            
            portfolio_values.append(portfolio_value)
    
    else:
        # 기존 로테이션 전략 로직
        for i, date in enumerate(df.index):
            season = seasons.loc[date] if date in seasons.index else np.nan
            
            if pd.isna(season):
                if i == 0:
                    portfolio_value = initial_capital
                    cash = initial_capital
                else:
                    if current_style and current_shares > 0:
                        portfolio_value = cash + (current_shares * df.loc[date, current_style])
                    else:
                        portfolio_value = cash
            else:
                target_style = strategy_rules[season]
                
                if i == 0:
                    current_style = target_style
                    current_price = df.loc[date, current_style]
                    current_shares = initial_capital / current_price
                    cash = 0
                    portfolio_value = initial_capital
                    
                    transactions.append({
                        'date': date,
                        'season': season,
                        'action': '초기투자',
                        'from_style': None,
                        'to_style': current_style,
                        'shares': current_shares,
                        'price': current_price,
                        'value': initial_capital
                    })
                
                elif target_style != current_style:
                    if current_style and current_shares > 0:
                        sell_price = df.loc[date, current_style]
                        cash = current_shares * sell_price
                        
                        transactions.append({
                            'date': date,
                            'season': season,
                            'action': '매도',
                            'from_style': current_style,
                            'to_style': None,
                            'shares': current_shares,
                            'price': sell_price,
                            'value': cash
                        })
                    
                    current_style = target_style
                    buy_price = df.loc[date, current_style]
                    current_shares = cash / buy_price
                    cash = 0
                    
                    transactions.append({
                        'date': date,
                        'season': season,
                        'action': '매수',
                        'from_style': None,
                        'to_style': current_style,
                        'shares': current_shares,
                        'price': buy_price,
                        'value': current_shares * buy_price
                    })
                    
                    portfolio_value = current_shares * buy_price
                
                else:
                    if current_style and current_shares > 0:
                        portfolio_value = current_shares * df.loc[date, current_style]
                    else:
                        portfolio_value = cash
                
                if i > 0:
                    prev_value = portfolio_values[-1]
                    period_return = (portfolio_value / prev_value) - 1
                    season_stats[season].append(period_return)
            
            portfolio_values.append(portfolio_value)
    
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    return portfolio_series, transactions, season_stats

def calculate_comprehensive_metrics(portfolio_series, initial_capital, season_stats):
    """포괄적인 성과 지표를 계산합니다."""
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    start_date = portfolio_series.index[0]
    end_date = portfolio_series.index[-1]
    investment_years = (end_date - start_date).days / 365.25
    
    cagr = (final_value / initial_capital) ** (1 / investment_years) - 1 if investment_years > 0 else 0
    
    cumulative_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / cumulative_max) - 1
    mdd = drawdown.min()
    
    returns = portfolio_series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(12)
    
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # 연도별 수익률 계산
    annual_returns = portfolio_series.resample('Y').last().pct_change().dropna()
    annual_performance = {}
    for year, ret in annual_returns.items():
        annual_performance[year.year] = ret
    
    # 최고/최저 연도 찾기
    best_year = annual_returns.idxmax().year if len(annual_returns) > 0 else None
    worst_year = annual_returns.idxmin().year if len(annual_returns) > 0 else None
    best_return = annual_returns.max() if len(annual_returns) > 0 else 0
    worst_return = annual_returns.min() if len(annual_returns) > 0 else 0
    
    # 양수/음수 연도 비율
    positive_years = len(annual_returns[annual_returns > 0])
    total_years = len(annual_returns)
    win_rate_annual = positive_years / total_years if total_years > 0 else 0
    
    season_performance = {}
    for season, returns_list in season_stats.items():
        if returns_list:
            season_performance[season] = {
                'avg_return': np.mean(returns_list),
                'win_rate': len([r for r in returns_list if r > 0]) / len(returns_list),
                'total_periods': len(returns_list)
            }
        else:
            season_performance[season] = {
                'avg_return': 0,
                'win_rate': 0,
                'total_periods': 0
            }
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'investment_years': investment_years,
        'season_performance': season_performance,
        'annual_performance': annual_performance,
        'best_year': best_year,
        'worst_year': worst_year,
        'best_return': best_return,
        'worst_return': worst_return,
        'win_rate_annual': win_rate_annual,
        'total_years': total_years
    }

def run_strategy_comparison(sp500_df, seasons, custom_strategy_name, custom_strategy_rules, initial_capital):
    """커스텀 전략과 기본 전략들을 비교합니다."""
    all_strategies = get_predefined_strategies()
    all_strategies[custom_strategy_name] = custom_strategy_rules
    
    results = {}
    
    print(f"\n=== 전략 비교 백테스팅 실행 ===")
    print(f"커스텀 전략: {custom_strategy_name}")
    print("기본 전략들과 성과를 비교합니다...\n")
    
    for strategy_name, strategy_rules in all_strategies.items():
        print(f"{strategy_name} 실행 중...")
        
        portfolio_series, transactions, season_stats = run_rotation_strategy(
            sp500_df, strategy_name, strategy_rules, seasons, initial_capital
        )
        
        metrics = calculate_comprehensive_metrics(portfolio_series, initial_capital, season_stats)
        
        results[strategy_name] = {
            'portfolio_series': portfolio_series,
            'transactions': transactions,
            'metrics': metrics,
            'strategy_rules': strategy_rules,
            'is_custom': strategy_name == custom_strategy_name
        }
        
        print(f"  최종 수익률: {metrics['total_return']:.2%}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  샤프 비율: {metrics['sharpe_ratio']:.2f}")
    
    return results

def create_enhanced_comparison_chart(results, custom_strategy_name, seasons):
    """커스텀 전략을 강조한 비교 차트를 생성합니다."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    
    # 커스텀 전략 강조를 위한 색상 설정
    colors = {}
    for i, strategy_name in enumerate(results.keys()):
        if strategy_name == custom_strategy_name:
            colors[strategy_name] = 'red'  # 커스텀 전략은 빨간색
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
    
    # 2. CAGR 비교 (커스텀 전략 강조)
    ax2 = axes[0, 1]
    strategies = list(results.keys())
    cagr_values = [results[s]['metrics']['cagr'] * 100 for s in strategies]
    bar_colors = [colors[s] for s in strategies]
    
    bars = ax2.bar(strategies, cagr_values, color=bar_colors, alpha=0.8)
    ax2.set_title('전략별 연평균 수익률 (CAGR)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('CAGR (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 커스텀 전략 막대에 테두리 추가
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
        if strategy == custom_strategy_name:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # 3. 리스크 조정 수익률 (샤프 비율)
    ax3 = axes[1, 0]
    sharpe_values = [results[s]['metrics']['sharpe_ratio'] for s in strategies]
    bars = ax3.bar(strategies, sharpe_values, color=bar_colors, alpha=0.8)
    ax3.set_title('전략별 샤프 비율', fontsize=14, fontweight='bold')
    ax3.set_ylabel('샤프 비율')
    ax3.tick_params(axis='x', rotation=45)
    
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
        if strategy == custom_strategy_name:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # 4. 최대 낙폭 비교
    ax4 = axes[1, 1]
    mdd_values = [abs(results[s]['metrics']['mdd']) * 100 for s in strategies]
    bars = ax4.bar(strategies, mdd_values, color=bar_colors, alpha=0.8)
    ax4.set_title('전략별 최대 낙폭 (MDD)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('MDD (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
        if strategy == custom_strategy_name:
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    
    # 5. 계절별 시장 분포
    ax5 = axes[2, 0]
    season_counts = seasons.value_counts()
    colors_season = {'여름': 'red', '봄': 'green', '가을': 'orange', '겨울': 'blue'}
    season_colors = [colors_season.get(season, 'gray') for season in season_counts.index]
    
    ax5.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', 
           colors=season_colors, startangle=90)
    ax5.set_title('RSI 기반 시장 계절 분포', fontsize=14, fontweight='bold')
    
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
        # 스타일명 단축
        short_style = style.replace('S&P500 ', '').replace('S&P 500 ', '').replace(' TR Index', '')
        ax6.text(0.1, y_pos, f"{season_icons[season]} {season}: {short_style}", 
                fontsize=12, transform=ax6.transAxes)
        y_pos -= 0.15
    
    # 7. 연도별 수익률 (커스텀 전략)
    ax7 = axes[3, 0]
    custom_metrics = results[custom_strategy_name]['metrics']
    annual_perf = custom_metrics['annual_performance']
    
    if annual_perf:
        years = list(annual_perf.keys())
        returns = [annual_perf[year] * 100 for year in years]
        bar_colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax7.bar(years, returns, color=bar_colors, alpha=0.7)
        ax7.set_title(f'{custom_strategy_name} 연도별 수익률', fontsize=14, fontweight='bold')
        ax7.set_ylabel('수익률 (%)')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax7.grid(True, alpha=0.3)
        
        # 회전된 연도 라벨
        ax7.tick_params(axis='x', rotation=45)
    
    # 8. 장기간 성과 요약
    ax8 = axes[3, 1]
    ax8.axis('off')
    ax8.set_title(f'{custom_strategy_name} 장기 성과 요약', fontsize=14, fontweight='bold')
    
    summary_text = [
        f"투자기간: {custom_metrics['investment_years']:.1f}년",
        f"총 수익률: {custom_metrics['total_return']:.1%}",
        f"연평균 수익률: {custom_metrics['cagr']:.2%}",
        f"최대 낙폭: {custom_metrics['mdd']:.1%}",
        f"변동성: {custom_metrics['volatility']:.1%}",
        f"샤프 비율: {custom_metrics['sharpe_ratio']:.2f}",
        f"연간 승률: {custom_metrics['win_rate_annual']:.1%}",
        f"최고 연도: {custom_metrics['best_year']} ({custom_metrics['best_return']:.1%})",
        f"최악 연도: {custom_metrics['worst_year']} ({custom_metrics['worst_return']:.1%})"
    ]
    
    y_pos = 0.9
    for text in summary_text:
        ax8.text(0.1, y_pos, text, fontsize=12, transform=ax8.transAxes)
        y_pos -= 0.1
    
    plt.tight_layout()
    plt.savefig(f'custom_strategy_comparison_{custom_strategy_name.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n전략 비교 차트가 저장되었습니다.")

def print_enhanced_results(results, custom_strategy_name):
    """커스텀 전략을 강조한 결과를 출력합니다."""
    print("\n" + "="*100)
    print("                   커스텀 vs 기본 전략 성과 비교")
    print("="*100)
    
    # 성과 요약 테이블
    print(f"{'전략명':<20} {'총수익률':<10} {'CAGR':<8} {'변동성':<8} {'MDD':<8} {'샤프비율':<8} {'연간승률':<8} {'구분':<8}")
    print("-" * 100)
    
    # 커스텀 전략을 먼저 출력
    for strategy_name, result in results.items():
        metrics = result['metrics']
        is_custom = result['is_custom']
        marker = "[커스텀]" if is_custom else "기본"
        
        print(f"{strategy_name:<20} "
              f"{metrics['total_return']:>8.1%} "
              f"{metrics['cagr']:>8.1%} "
              f"{metrics['volatility']:>8.1%} "
              f"{metrics['mdd']:>8.1%} "
              f"{metrics['sharpe_ratio']:>8.2f} "
              f"{metrics['win_rate_annual']:>8.1%} "
              f"{marker:<8}")
        
        if is_custom:  # 커스텀 전략 뒤에 구분선
            print("-" * 100)
    
    print("="*100)
    
    # 커스텀 전략 상세 분석
    custom_metrics = results[custom_strategy_name]['metrics']
    
    print(f"\n=== '{custom_strategy_name}' 상세 성과 분석 ===")
    print(f"투자 기간: {custom_metrics['investment_years']:.1f}년")
    print(f"초기 → 최종: {custom_metrics['total_return']:+.1%} (연평균 {custom_metrics['cagr']:+.2%})")
    print(f"리스크 지표: MDD {custom_metrics['mdd']:.1%}, 변동성 {custom_metrics['volatility']:.1%}")
    print(f"샤프 비율: {custom_metrics['sharpe_ratio']:.2f}")
    print(f"연간 성과: {custom_metrics['total_years']}년 중 {int(custom_metrics['win_rate_annual'] * custom_metrics['total_years'])}년 양수 수익 (승률 {custom_metrics['win_rate_annual']:.1%})")
    
    if custom_metrics.get('best_year') and custom_metrics.get('worst_year'):
        print(f"최고 연도: {custom_metrics['best_year']}년 ({custom_metrics['best_return']:+.1%})")
        print(f"최악 연도: {custom_metrics['worst_year']}년 ({custom_metrics['worst_return']:+.1%})")
    
    # 순위 분석
    rankings = {
        'total_return': sorted(results.items(), key=lambda x: x[1]['metrics']['total_return'], reverse=True),
        'cagr': sorted(results.items(), key=lambda x: x[1]['metrics']['cagr'], reverse=True),
        'sharpe_ratio': sorted(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'], reverse=True),
        'mdd': sorted(results.items(), key=lambda x: abs(x[1]['metrics']['mdd']))  # MDD는 낮을수록 좋음
    }
    
    print(f"\n=== '{custom_strategy_name}' 순위 분석 ===")
    for metric, ranked_list in rankings.items():
        for rank, (strategy, _) in enumerate(ranked_list, 1):
            if strategy == custom_strategy_name:
                metric_names = {
                    'total_return': '총수익률',
                    'cagr': 'CAGR',
                    'sharpe_ratio': '샤프비율',
                    'mdd': '리스크관리(MDD)'
                }
                print(f"{metric_names[metric]}: {rank}위 / {len(results)}개 전략")
                break
    
    # 계절별 성과
    print(f"\n=== '{custom_strategy_name}' 계절별 성과 ===")
    season_perf = custom_metrics['season_performance']
    season_icons = {'봄': '[봄]', '여름': '[여름]', '가을': '[가을]', '겨울': '[겨울]'}
    
    print(f"{'계절':<8} {'평균수익률':<12} {'승률':<8} {'거래횟수':<8}")
    print("-" * 40)
    for season in ['봄', '여름', '가을', '겨울']:
        perf = season_perf[season]
        icon = season_icons[season]
        print(f"{icon} {season:<6} {perf['avg_return']:>10.2%} "
              f"{perf['win_rate']:>8.1%} {perf['total_periods']:>8d}")
    
    # 연도별 수익률 (최근 10년)
    if custom_metrics.get('annual_performance'):
        annual_perf = custom_metrics['annual_performance']
        recent_years = sorted(annual_perf.keys())[-10:]  # 최근 10년
        
        print(f"\n=== '{custom_strategy_name}' 최근 연도별 수익률 ===")
        print("연도  수익률     연도  수익률")
        print("-" * 25)
        
        for i in range(0, len(recent_years), 2):
            year1 = recent_years[i]
            ret1 = annual_perf[year1]
            
            if i + 1 < len(recent_years):
                year2 = recent_years[i + 1]
                ret2 = annual_perf[year2]
                print(f"{year1}  {ret1:>+6.1%}     {year2}  {ret2:>+6.1%}")
            else:
                print(f"{year1}  {ret1:>+6.1%}")

def get_user_input():
    """사용자로부터 백테스팅 파라미터를 입력받습니다."""
    print("\n=== 커스텀 RSI 기반 스타일 로테이션 전략 백테스팅 ===")
    
    try:
        start_year = int(input("\n백테스팅 시작 연도 (YYYY): "))
        start_month = int(input("백테스팅 시작 월 (1-12): "))
        end_year = int(input("백테스팅 종료 연도 (YYYY): "))
        end_month = int(input("백테스팅 종료 월 (1-12): "))
        initial_capital = float(input("초기 투자 원금 (원): "))
        
        return {
            'start_year': start_year,
            'start_month': start_month,
            'end_year': end_year,
            'end_month': end_month,
            'initial_capital': initial_capital
        }
    
    except (ValueError, KeyboardInterrupt) as e:
        print(f"입력 오류: {e}")
        return None

def main():
    """메인 함수 - 커스텀 RSI 기반 로테이션 전략 백테스팅을 실행합니다."""
    # 1. 데이터 로딩
    import os
    # 스크립트 파일이 있는 디렉토리로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sp500_df = load_sp500_data(os.path.join(script_dir, 'sp500_data.xlsx'))
    if sp500_df is None:
        return
    
    rsi_series = load_rsi_data(os.path.join(script_dir, 'RSI_DATE.xlsx'))
    if rsi_series is None:
        return
    
    # 2. 커스텀 전략 설정
    custom_result = get_custom_strategy()
    if custom_result is None:
        return
    
    custom_strategy_name, custom_strategy_rules = custom_result
    
    # 3. 전략 저장 옵션
    save_option = input(f"\n'{custom_strategy_name}' 전략을 저장하시겠습니까? (y/n): ").lower()
    if save_option == 'y':
        save_custom_strategy(custom_strategy_name, custom_strategy_rules)
    
    # 4. 사용자 입력
    user_input = get_user_input()
    if user_input is None:
        return
    
    try:
        # 5. 데이터 정렬 및 필터링
        start_date = pd.Timestamp(user_input['start_year'], user_input['start_month'], 1)
        # 해당 월의 마지막 날로 설정
        end_date = pd.Timestamp(user_input['end_year'], user_input['end_month'], 1) + pd.offsets.MonthEnd(0)
        
        sp500_aligned, rsi_aligned = align_data(sp500_df, rsi_series, start_date, end_date)
        
        # 6. 계절 분류
        seasons = classify_market_season(rsi_aligned)
        valid_seasons = seasons.dropna()
        
        if len(valid_seasons) == 0:
            print("경고: 계절 분류된 데이터가 없습니다.")
            return
        
        print(f"계절 분류 완료: {len(valid_seasons)}개 데이터 포인트")
        
        # 7. 전략 비교 실행
        results = run_strategy_comparison(
            sp500_aligned, seasons, custom_strategy_name, 
            custom_strategy_rules, user_input['initial_capital']
        )
        
        # 8. 결과 출력
        print_enhanced_results(results, custom_strategy_name)
        
        # 9. 시각화
        create_enhanced_comparison_chart(results, custom_strategy_name, valid_seasons)
        
        print(f"\n[SUCCESS] '{custom_strategy_name}' 전략 분석이 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()