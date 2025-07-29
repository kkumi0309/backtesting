"""
월별 전처리된 금융 데이터 분석 예시
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from financial_data_collector import FinancialDataCollector
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def monthly_data_example():
    """월별 전처리된 데이터 수집 및 분석 예시"""
    print("=== 월별 전처리 데이터 수집 예시 ===")
    
    # 데이터 수집기 초기화 (2015-2024)
    collector = FinancialDataCollector(start_date='2015-01-01', end_date='2024-12-31')
    
    # 월별 전처리된 데이터 수집
    monthly_data = collector.collect_all_data(monthly_processing=True)
    
    if monthly_data is not None:
        print("\n=== 월별 전처리 데이터 확인 ===")
        print(f"데이터 형태: {monthly_data.shape}")
        print(f"기간: {monthly_data.index.min().strftime('%Y-%m')} ~ {monthly_data.index.max().strftime('%Y-%m')}")
        print(f"변수: {list(monthly_data.columns)}")
        
        print("\n=== 최근 12개월 데이터 ===")
        print(monthly_data.tail(12))
        
        print("\n=== 요약 통계 ===")
        print(monthly_data.describe())
        
        # CSV 저장
        collector.combined_data = monthly_data
        collector.save_data('monthly_financial_data.csv')
        
        return monthly_data
    
    return None

def correlation_analysis(monthly_data):
    """상관관계 분석"""
    if monthly_data is None:
        return
    
    print("\n=== 상관관계 분석 ===")
    corr_matrix = monthly_data.corr()
    print(corr_matrix)
    
    # 상관관계 히트맵
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', square=True, linewidths=0.5, 
                cbar_kws={'shrink': 0.8})
    plt.title('월별 금융 데이터 상관관계 히트맵', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('monthly_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def time_series_plots(monthly_data):
    """시계열 플롯"""
    if monthly_data is None:
        return
    
    print("\n=== 시계열 플롯 생성 ===")
    
    # 자산 수익률 플롯
    asset_returns = [col for col in monthly_data.columns if 'Returns' in col]
    if asset_returns:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(asset_returns, 1):
            plt.subplot(2, 2, i)
            plt.plot(monthly_data.index, monthly_data[col], linewidth=1.5)
            plt.title(f'{col}', fontsize=12)
            plt.ylabel('수익률 (%)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('monthly_asset_returns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 금리/스프레드 변화량 플롯
    rate_changes = [col for col in monthly_data.columns if ('Change' in col or 'BP' in col)]
    if rate_changes:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(rate_changes, 1):
            if i <= 4:  # 최대 4개까지만 플롯
                plt.subplot(2, 2, i)
                plt.plot(monthly_data.index, monthly_data[col], linewidth=1.5, color='red')
                plt.title(f'{col}', fontsize=12)
                if 'BP' in col:
                    plt.ylabel('변화량 (bp)')
                else:
                    plt.ylabel('변화량')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('monthly_rate_changes.png', dpi=300, bbox_inches='tight')
        plt.show()

def risk_return_analysis(monthly_data):
    """위험-수익률 분석"""
    if monthly_data is None:
        return
    
    print("\n=== 위험-수익률 분석 ===")
    
    # 자산 수익률만 선택
    asset_returns = monthly_data[[col for col in monthly_data.columns if 'Returns' in col]]
    
    if not asset_returns.empty:
        # 평균 수익률과 변동성 계산
        mean_returns = asset_returns.mean() * 12  # 연율화
        volatility = asset_returns.std() * np.sqrt(12)  # 연율화
        
        risk_return_df = pd.DataFrame({
            '연평균 수익률 (%)': mean_returns,
            '연변동성 (%)': volatility,
            '샤프비율 (위험무료수익률=0 가정)': mean_returns / volatility
        })
        
        print(risk_return_df)
        
        # 위험-수익률 산점도
        plt.figure(figsize=(10, 8))
        for i, asset in enumerate(asset_returns.columns):
            plt.scatter(volatility.iloc[i], mean_returns.iloc[i], s=100, alpha=0.7)
            plt.annotate(asset.replace('_Returns', ''), 
                        (volatility.iloc[i], mean_returns.iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('연변동성 (%)')
        plt.ylabel('연평균 수익률 (%)')
        plt.title('위험-수익률 관계')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('risk_return_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return risk_return_df
    
    return None

def descriptive_statistics(monthly_data):
    """기술통계 분석"""
    if monthly_data is None:
        return
    
    print("\n=== 기술통계 분석 ===")
    
    stats_df = monthly_data.describe()
    
    # 추가 통계량 계산
    additional_stats = pd.DataFrame({
        '왜도(Skewness)': monthly_data.skew(),
        '첨도(Kurtosis)': monthly_data.kurtosis(),
        '최솟값 날짜': monthly_data.idxmin(),
        '최댓값 날짜': monthly_data.idxmax()
    })
    
    print("\n기본 통계량:")
    print(stats_df)
    
    print("\n추가 통계량:")
    print(additional_stats)
    
    # 분포 히스토그램
    asset_returns = [col for col in monthly_data.columns if 'Returns' in col]
    if asset_returns:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(asset_returns, 1):
            plt.subplot(2, 2, i)
            plt.hist(monthly_data[col], bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'{col} 분포')
            plt.xlabel('수익률 (%)')
            plt.ylabel('빈도')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('return_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

# 실행
if __name__ == "__main__":
    # 월별 데이터 수집
    monthly_data = monthly_data_example()
    
    if monthly_data is not None:
        # 각종 분석 수행
        correlation_analysis(monthly_data)
        time_series_plots(monthly_data)
        risk_return_stats = risk_return_analysis(monthly_data)
        descriptive_statistics(monthly_data)
        
        print("\n=== 분석 완료 ===")
        print("생성된 파일:")
        print("- monthly_financial_data.csv")
        print("- monthly_correlation_heatmap.png")
        print("- monthly_asset_returns.png")
        print("- monthly_rate_changes.png")
        print("- risk_return_plot.png")
        print("- return_distributions.png")