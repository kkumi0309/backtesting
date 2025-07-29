import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, window=14):
    """RSI 계산 함수"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_sp500_style_data_from_api():
    """yfinance를 통해 S&P 500 스타일 지수 데이터를 로드합니다."""
    try:
        # S&P 500 스타일 지수 티커들
        style_tickers = {
            'S&P500 Growth': 'IVW',           # iShares Core S&P 500 Growth ETF
            'S&P500 Value': 'IVE',            # iShares Core S&P 500 Value ETF  
            'S&P500 Momentum': 'MTUM',        # iShares MSCI USA Momentum Factor ETF
            'S&P500 Quality': 'QUAL',         # iShares MSCI USA Quality Factor ETF
            'S&P500 Low Volatiltiy Index': 'USMV',  # iShares MSCI USA Min Vol Factor ETF
            'S&P500 Div Aristocrt TR Index': 'NOBL'  # ProShares S&P 500 Dividend Aristocrats ETF
        }
        
        print("API를 통해 S&P 500 스타일 지수 데이터를 다운로드 중...")
        
        # 데이터 다운로드 (2000-01-01부터 현재까지)
        start_date = '2000-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        style_data = {}
        for style_name, ticker in style_tickers.items():
            try:
                # Ticker 객체 사용으로 변경
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Close 컬럼 사용 (Ticker.history()는 단순한 컬럼명 사용)
                    monthly_data = data['Close'].resample('M').last()
                    style_data[style_name] = monthly_data
                    print(f"  * {style_name} ({ticker}): {len(monthly_data)}개 데이터")
                else:
                    print(f"  x {style_name} ({ticker}): 데이터 없음")
                    
            except Exception as e:
                print(f"  x {style_name} ({ticker}): 오류 - {e}")
        
        if not style_data:
            raise ValueError("스타일 지수 데이터를 가져올 수 없습니다.")
        
        # DataFrame으로 결합
        df = pd.DataFrame(style_data)
        
        # 결측값 처리 (앞의 값으로 채움)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"S&P 500 스타일 지수 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
        return df
        
    except Exception as e:
        print(f"API 데이터 로딩 오류: {e}")
        print("로컬 파일을 사용하세요.")
        return None

def load_sp500_rsi_from_api():
    """S&P 500 지수로부터 RSI를 계산합니다."""
    try:
        print("S&P 500 지수 데이터 다운로드 중...")
        
        # S&P 500 지수 다운로드
        start_date = '1999-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Ticker 객체 사용
        sp500_ticker = yf.Ticker('^GSPC')
        sp500_data = sp500_ticker.history(start=start_date, end=end_date)
        
        if sp500_data.empty:
            raise ValueError("S&P 500 지수 데이터를 가져올 수 없습니다.")
        
        # 월별 종가 데이터
        monthly_close = sp500_data['Close'].resample('M').last()
        
        # RSI 계산 (14개월 기준)
        rsi_values = calculate_rsi(monthly_close, window=14)
        
        # 결측값 제거
        rsi_series = rsi_values.dropna()
        
        print(f"RSI 데이터 계산 완료: {len(rsi_series)}개 데이터 포인트")
        return rsi_series
        
    except Exception as e:
        print(f"RSI 계산 오류: {e}")
        print("로컬 파일을 사용하세요.")
        return None

def load_sp500_data_with_fallback(file_path=None):
    """API 우선, 실패 시 로컬 파일 사용"""
    # 먼저 API 시도
    sp500_df = load_sp500_style_data_from_api()
    
    if sp500_df is not None:
        return sp500_df
    
    # API 실패 시 로컬 파일 사용
    if file_path:
        try:
            df = pd.read_excel(file_path)
            date_column = df.columns[0]
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
            print(f"로컬 S&P 500 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 지수")
            return df
        except Exception as e:
            print(f"로컬 S&P 500 데이터 로딩 오류: {e}")
            return None
    
    return None

def load_rsi_data_with_fallback(file_path=None):
    """API 우선, 실패 시 로컬 파일 사용"""
    # 먼저 API 시도  
    rsi_series = load_sp500_rsi_from_api()
    
    if rsi_series is not None:
        return rsi_series
    
    # API 실패 시 로컬 파일 사용
    if file_path:
        try:
            df = pd.read_excel(file_path, skiprows=1)
            date_column = df.columns[0]
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
            rsi_series = df['RSI'].dropna()
            print(f"로컬 RSI 데이터 로딩 완료: {len(rsi_series)}개 데이터 포인트")
            return rsi_series
        except Exception as e:
            print(f"로컬 RSI 데이터 로딩 오류: {e}")
            return None
    
    return None

if __name__ == "__main__":
    # 테스트
    print("=== API 데이터 로더 테스트 ===")
    
    # S&P 500 스타일 지수 테스트
    sp500_df = load_sp500_style_data_from_api()
    if sp500_df is not None:
        print(f"\nS&P 500 스타일 지수 미리보기:")
        print(sp500_df.head())
        print(f"최신 데이터: {sp500_df.index[-1]}")
    
    # RSI 테스트
    rsi_series = load_sp500_rsi_from_api()
    if rsi_series is not None:
        print(f"\nRSI 데이터 미리보기:")
        print(rsi_series.head())
        print(f"최신 RSI: {float(rsi_series.iloc[-1]):.2f}")
        print(f"최신 데이터: {rsi_series.index[-1]}")