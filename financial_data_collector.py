# -*- coding: utf-8 -*-
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

# Windows 콘솔 출력 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class FinancialDataCollector:
    """
    학술 연구를 위한 금융 및 경제 시계열 데이터 수집기
    """
    
    def __init__(self, start_date='2010-01-01', end_date=None):
        """
        초기화
        
        Parameters:
        start_date (str): 시작 날짜 (YYYY-MM-DD 형식)
        end_date (str): 종료 날짜 (기본값: 오늘 날짜)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
        self.data = {}
        
    def collect_sp500(self):
        """S&P 500 지수 데이터 수집"""
        try:
            print("S&P 500 지수 데이터 수집 중...")
            sp500 = pdr.get_data_yahoo('^GSPC', self.start_date, self.end_date)
            self.data['SP500'] = sp500['Adj Close']
            print("✓ S&P 500 데이터 수집 완료")
            return sp500['Adj Close']
        except Exception as e:
            print(f"✗ S&P 500 데이터 수집 실패: {e}")
            return None
    
    def collect_treasury_10y(self):
        """미국 10년물 국채 금리 데이터 수집"""
        try:
            print("미국 10년물 국채 금리 데이터 수집 중...")
            treasury_10y = pdr.get_data_fred('DGS10', self.start_date, self.end_date)
            treasury_10y = treasury_10y.dropna()
            self.data['Treasury_10Y'] = treasury_10y['DGS10']
            print("✓ 10년물 국채 금리 데이터 수집 완료")
            return treasury_10y['DGS10']
        except Exception as e:
            print(f"✗ 10년물 국채 금리 데이터 수집 실패: {e}")
            return None
    
    def collect_sp_gsci(self):
        """S&P GSCI 원자재지수 데이터 수집 (Yahoo Finance 사용)"""
        try:
            print("S&P GSCI 원자재지수 데이터 수집 중...")
            # Yahoo Finance의 S&P GSCI Ticker: ^SPGSCI
            gsci = pdr.get_data_yahoo('^SPGSCI', self.start_date, self.end_date)
            gsci = gsci.dropna(subset=['Adj Close'])
            self.data['SP_GSCI'] = gsci['Adj Close']
            print("✓ S&P GSCI 원자재지수 데이터 수집 완료")
            return gsci['Adj Close']
        except Exception as e:
            print(f"✗ S&P GSCI 데이터 수집 실패: {e}")
            print("대안으로 원자재 ETF(DJP) 데이터 수집을 시도합니다.")
            try:
                djp = pdr.get_data_yahoo('DJP', self.start_date, self.end_date)
                self.data['SP_GSCI'] = djp['Adj Close']
                print("✓ 원자재 ETF(DJP) 데이터 수집 완료")
                return djp['Adj Close']
            except Exception as e2:
                print(f"✗ 원자재 ETF(DJP) 데이터 수집도 실패: {e2}")
                return None
    
    def collect_gold_price(self):
        """금 가격 데이터 수집"""
        try:
            print("금 가격 데이터 수집 중...")
            gold = pdr.get_data_fred('GOLDPMGBD228NLBM', self.start_date, self.end_date)
            gold = gold.dropna()
            self.data['Gold'] = gold['GOLDPMGBD228NLBM']
            print("✓ 금 가격 데이터 수집 완료")
            return gold['GOLDPMGBD228NLBM']
        except Exception as e:
            print(f"✗ 금 가격 데이터 수집 실패: {e}")
            return None
    
    def collect_vix(self):
        """VIX 지수 데이터 수집"""
        try:
            print("VIX 지수 데이터 수집 중...")
            vix = pdr.get_data_fred('VIXCLS', self.start_date, self.end_date)
            vix = vix.dropna()
            self.data['VIX'] = vix['VIXCLS']
            print("✓ VIX 지수 데이터 수집 완료")
            return vix['VIXCLS']
        except Exception as e:
            print(f"✗ VIX 지수 데이터 수집 실패: {e}")
            return None
    
    def collect_yield_spread_10y2y(self):
        """10년물-2년물 금리차 데이터 수집"""
        try:
            print("10년물-2년물 금리차 데이터 수집 중...")
            yield_spread = pdr.get_data_fred('T10Y2Y', self.start_date, self.end_date)
            yield_spread = yield_spread.dropna()
            self.data['Yield_Spread_10Y2Y'] = yield_spread['T10Y2Y']
            print("✓ 장단기 금리차 데이터 수집 완료")
            return yield_spread['T10Y2Y']
        except Exception as e:
            print(f"✗ 장단기 금리차 데이터 수집 실패: {e}")
            return None
    
    def collect_credit_spread(self):
        """BAA등급 회사채-10년물 신용 스프레드 데이터 수집"""
        try:
            print("신용 스프레드 데이터 수집 중...")
            credit_spread = pdr.get_data_fred('BAA10Y', self.start_date, self.end_date)
            credit_spread = credit_spread.dropna()
            self.data['Credit_Spread'] = credit_spread['BAA10Y']
            print("✓ 신용 스프레드 데이터 수집 완료")
            return credit_spread['BAA10Y']
        except Exception as e:
            print(f"✗ 신용 스프레드 데이터 수집 실패: {e}")
            return None
    
    def collect_usdx(self):
        """미국 달러 인덱스(Trade Weighted U.S. Dollar Index) 데이터 수집"""
        try:
            print("미국 달러 인덱스 데이터 수집 중...")
            # FRED Ticker for Trade Weighted U.S. Dollar Index: Broad, Goods and Services
            usdx = pdr.get_data_fred('DTWEXBGS', self.start_date, self.end_date)
            usdx = usdx.dropna()
            self.data['USDX'] = usdx['DTWEXBGS']
            print("✓ 미국 달러 인덱스 데이터 수집 완료")
            return usdx['DTWEXBGS']
        except Exception as e:
            print(f"✗ 미국 달러 인덱스 데이터 수집 실패: {e}")
            return None
    
    def preprocess_to_monthly(self, raw_data):
        """원시 데이터를 월별 주기로 전처리"""
        if raw_data is None or raw_data.empty:
            return None
        
        print("\n=== 월별 데이터 전처리 시작 ===")
        
        monthly_data = {}
        
        # 자산 가격/지수 → 월별 수익률(%) 변환
        asset_columns = ['SP500', 'Gold', 'SP_GSCI', 'USDX']
        korean_names = {'SP500': 'SP500_수익률', 'Gold': '금_수익률', 'SP_GSCI': '원자재_수익률', 'USDX': '달러인덱스_수익률'}
        for col in asset_columns:
            if col in raw_data.columns:
                print(f"{col} → 월별 수익률(%) 변환 중...")
                monthly_series = raw_data[col].resample('M').last()  # 월말 종가
                monthly_returns = monthly_series.pct_change() * 100  # 백분율 변환
                monthly_data[korean_names[col]] = monthly_returns
                print(f"✓ {col} 월별 수익률 변환 완료")
        
        # 금리/스프레드/VIX → 월별 변화량(basis point) 변환
        rate_columns = ['Treasury_10Y', 'VIX', 'Yield_Spread_10Y2Y', 'Credit_Spread']
        rate_korean_names = {
            'Treasury_10Y': '미국채10년_변화BP', 
            'VIX': 'VIX_변화', 
            'Yield_Spread_10Y2Y': '장단기금리차_변화BP', 
            'Credit_Spread': '신용스프레드_변화BP'
        }
        for col in rate_columns:
            if col in raw_data.columns:
                print(f"{col} → 월별 변화량(bp) 변환 중...")
                monthly_series = raw_data[col].resample('M').last()  # 월말 값
                if col == 'VIX':
                    monthly_changes = monthly_series.diff()  # VIX는 그대로
                    monthly_data[rate_korean_names[col]] = monthly_changes
                else:
                    monthly_changes = monthly_series.diff() * 100  # basis point 변환
                    monthly_data[rate_korean_names[col]] = monthly_changes
                print(f"✓ {col} 월별 변화량 변환 완료")
        
        # 전처리된 데이터를 DataFrame으로 통합
        if monthly_data:
            processed_df = pd.DataFrame(monthly_data)
            
            # 결측치가 포함된 행 완전 제거
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna()
            final_rows = len(processed_df)
            
            print(f"\n✓ 월별 전처리 완료")
            print(f"  - 결측치 제거: {initial_rows} → {final_rows} 행 ({initial_rows - final_rows}개 행 제거)")
            print(f"  - 최종 형태: {processed_df.shape}")
            print(f"  - 기간: {processed_df.index.min().strftime('%Y-%m')} ~ {processed_df.index.max().strftime('%Y-%m')}")
            print(f"  - 변수: {list(processed_df.columns)}")
            
            return processed_df
        else:
            print("✗ 전처리할 데이터가 없습니다.")
            return None
    
    def collect_all_data(self, monthly_processing=True):
        """모든 데이터를 수집하고 통합된 DataFrame 반환"""
        print("=== 금융 및 경제 데이터 수집 시작 ===")
        print(f"수집 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        print("")
        
        # 각 데이터 수집
        self.collect_sp500()
        self.collect_treasury_10y()
        self.collect_sp_gsci()
        self.collect_gold_price()
        self.collect_vix()
        self.collect_yield_spread_10y2y()
        self.collect_credit_spread()
        self.collect_usdx()
        
        print("\n=== 원시 데이터 통합 ===")
        
        # 데이터가 있는 경우만 DataFrame으로 결합
        if self.data:
            raw_df = pd.DataFrame()
            for name, series in self.data.items():
                if series is not None and not series.empty:
                    raw_df[name] = series
            
            if not raw_df.empty:
                print(f"✓ 원시 데이터 통합 완료")
                print(f"  - 형태: {raw_df.shape}")
                print(f"  - 기간: {raw_df.index.min().date()} ~ {raw_df.index.max().date()}")
                print(f"  - 변수: {list(raw_df.columns)}")
                
                # 월별 전처리 수행
                if monthly_processing:
                    return self.preprocess_to_monthly(raw_df)
                else:
                    # 원 데이터에 한글 컬럼명 적용
                    korean_column_names = {
                        'SP500': 'SP500지수',
                        'Treasury_10Y': '미국채10년금리',
                        'SP_GSCI': '원자재지수',
                        'Gold': '금가격',
                        'VIX': 'VIX지수',
                        'Yield_Spread_10Y2Y': '장단기금리차',
                        'Credit_Spread': '신용스프레드'
                        , 'USDX': '달러인덱스'
                    }
                    raw_df.rename(columns=korean_column_names, inplace=True)
                    return raw_df
            else:
                print("✗ 수집된 데이터가 없습니다.")
                return None
        else:
            print("✗ 수집된 데이터가 없습니다.")
            return None
    
    def save_data(self, filename='financial_data.csv'):
        """데이터를 CSV 파일로 저장"""
        if hasattr(self, 'combined_data') and self.combined_data is not None:
            filepath = f"C:\\Users\\wnghk\\Desktop\\ACADEMY\\2025-1.5\\계량경제\\{filename}"
            self.combined_data.to_csv(filepath)
            print(f"✓ 데이터가 {filepath}에 저장되었습니다.")
        else:
            print("✗ 저장할 데이터가 없습니다. collect_all_data()를 먼저 실행해주세요.")
    
    def get_summary_stats(self, data=None):
        """데이터 요약 통계 출력"""
        if data is None:
            data = getattr(self, 'combined_data', None)
        
        if data is not None:
            print("\n=== 데이터 요약 통계 ===")
            print(data.describe())
            print("\n=== 결측치 현황 ===")
            print(data.isnull().sum())
        else:
            print("✗ 분석할 데이터가 없습니다.")

# 사용 예시
if __name__ == "__main__":
    # 데이터 수집기 초기화 (1990년부터 데이터)
    collector = FinancialDataCollector(start_date='1990-01-01')
    
    # 모든 데이터 수집 (원 데이터로)
    financial_data = collector.collect_all_data(monthly_processing=False)
    
    if financial_data is not None:
        # 요약 통계 출력
        collector.combined_data = financial_data
        collector.get_summary_stats(financial_data)
        
        # CSV 파일로 저장
        collector.save_data('financial_data_raw.csv')
        
        print("\n=== 최근 5개 데이터 ===")
        print(financial_data.tail())