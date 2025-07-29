"""
금융 데이터 수집기 사용 예시 및 고급 분석
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from financial_data_collector import FinancialDataCollector
import numpy as np

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def basic_usage_example():
    """기본 사용법 예시"""
    print("=== 기본 사용법 예시 ===")
    
    # 1. 데이터 수집기 초기화
    collector = FinancialDataCollector(start_date='2020-01-01', end_date='2024-12-31')
    
    # 2. 모든 데이터 수집
    data = collector.collect_all_data()
    
    if data is not None:
        # 3. 기본 정보 확인
        print(f"\n데이터 형태: {data.shape}")
        print(f"수집 기간: {data.index.min().date()} ~ {data.index.max().date()}")
        
        # 4. 요약 통계
        collector.combined_data = data
        collector.get_summary_stats(data)
        
        # 5. 데이터 저장
        collector.save_data('financial_data_2020_2024.csv')
        
        return data
    return None

def advanced_analysis_example(data):
    """고급 분석 예시"""
    if data is None:
        print("분석할 데이터가 없습니다.")
        return
    
    print("\n=== 고급 분석 예시 ===")
    
    # 1. 상관관계 분석
    print("\n1. 상관관계 분석")
    correlation_matrix = data.corr()
    print(correlation_matrix)
    
    # 2. 상관관계 히트맵 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.3f', square=True, cbar_kws={'shrink': 0.8})
    plt.title('금융 데이터 상관관계 히트맵', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 시계열 플롯
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, column in enumerate(data.columns):
        if i < len(axes):
            axes[i].plot(data.index, data[column], linewidth=1.5)
            axes[i].set_title(f'{column}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
    
    # 남는 subplot 제거
    for j in range(len(data.columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('time_series_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 수익률 계산 및 분석
    print("\n4. 수익률 분석 (S&P 500 기준)")
    if 'SP500' in data.columns:
        returns = data['SP500'].pct_change().dropna()
        
        print(f"평균 일일 수익률: {returns.mean():.4f}")
        print(f"일일 변동성: {returns.std():.4f}")
        print(f"연간 수익률 (추정): {returns.mean() * 252:.4f}")
        print(f"연간 변동성 (추정): {returns.std() * np.sqrt(252):.4f}")
        
        # 수익률 분포 시각화
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(returns * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('S&P 500 일일 수익률 분포')
        plt.xlabel('수익률 (%)')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(returns.index, returns.cumsum() * 100)
        plt.title('S&P 500 누적 수익률')
        plt.xlabel('날짜')
        plt.ylabel('누적 수익률 (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('returns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def custom_date_analysis():
    """특정 기간 분석 예시"""
    print("\n=== 특정 기간 분석 예시 ===")
    
    # 코로나19 시기 분석 (2020-2022)
    collector_covid = FinancialDataCollector(start_date='2020-01-01', end_date='2022-12-31')
    covid_data = collector_covid.collect_all_data()
    
    if covid_data is not None:
        print("\n코로나19 시기 (2020-2022) 데이터:")
        print(covid_data.describe())
        
        # VIX와 S&P 500 관계 분석
        if 'VIX' in covid_data.columns and 'SP500' in covid_data.columns:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(covid_data['VIX'], covid_data['SP500'].pct_change() * 100, alpha=0.6)
            plt.xlabel('VIX (변동성 지수)')
            plt.ylabel('S&P 500 일일 수익률 (%)')
            plt.title('VIX vs S&P 500 수익률 관계')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            covid_data[['VIX', 'SP500']].plot(secondary_y='SP500', figsize=(6, 4))
            plt.title('VIX vs S&P 500 시계열')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('vix_sp500_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

def individual_data_collection():
    """개별 데이터 수집 예시"""
    print("\n=== 개별 데이터 수집 예시 ===")
    
    collector = FinancialDataCollector(start_date='2023-01-01')
    
    # 개별 데이터 수집
    sp500 = collector.collect_sp500()
    vix = collector.collect_vix()
    gold = collector.collect_gold_price()
    
    # 특정 데이터만 결합
    if all(data is not None for data in [sp500, vix, gold]):
        custom_data = pd.DataFrame({
            'S&P500': sp500,
            'VIX': vix,
            'Gold': gold
        }).dropna()
        
        print(f"\n맞춤 데이터셋 생성 완료: {custom_data.shape}")
        print(custom_data.head())
        
        # 맞춤 분석
        print(f"\n상관관계:")
        print(custom_data.corr())

# 실행
if __name__ == "__main__":
    # 기본 사용법
    data = basic_usage_example()
    
    # 고급 분석
    if data is not None:
        advanced_analysis_example(data)
    
    # 특정 기간 분석
    custom_date_analysis()
    
    # 개별 데이터 수집
    individual_data_collection()