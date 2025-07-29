
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# --- 기본 설정 ---
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

app = FastAPI()

# --- 데이터 로딩 (앱 시작 시 한 번만 실행) ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

sp500_df = None
rsi_series = None

@app.on_event("startup")
def load_data():
    global sp500_df, rsi_series
    # In a real web app, you'd use a more robust way to manage file paths.
    # For this example, we assume the data files are in the same directory.
    try:
        sp500_df = pd.read_excel("sp500_data.xlsx", engine='openpyxl')
        date_column = sp500_df.columns[0]
        sp500_df[date_column] = pd.to_datetime(sp500_df[date_column])
        sp500_df.set_index(date_column, inplace=True)
        sp500_df = sp500_df[sp500_df.index.notna()]
        sp500_df.sort_index(inplace=True)
        print("S&P 500 데이터 로딩 완료")

        rsi_df = pd.read_excel("RSI_DATE.xlsx", skiprows=1, engine='openpyxl')
        date_column_rsi = rsi_df.columns[0]
        rsi_df[date_column_rsi] = pd.to_datetime(rsi_df[date_column_rsi])
        rsi_df.set_index(date_column_rsi, inplace=True)
        rsi_df.sort_index(inplace=True)
        rsi_series = rsi_df['RSI'].dropna()
        print("RSI 데이터 로딩 완료")
    except FileNotFoundError as e:
        print(f"데이터 파일 로딩 오류: {e}. 'backend' 폴더에 sp500_data.xlsx와 RSI_DATE.xlsx 파일이 있는지 확인하세요.")
        # In a real app, you might want to exit or handle this more gracefully
        sys.exit(1)


# --- Pydantic 모델 (API 요청/응답 데이터 형식 정의) ---
class Strategy(BaseModel):
    봄: str
    여름: str
    가을: str
    겨울: str

class SpecialStrategy(BaseModel):
    봄: str
    가을: str

class BacktestRequest(BaseModel):
    strategy_name: str
    strategy_rules: Strategy
    special_strategy: Optional[SpecialStrategy] = None

# --- 백테스팅 핵심 로직 (기존 코드 재활용 및 수정) ---
RSI_THRESHOLDS = {'summer': 70, 'winter': 30}

def classify_market_season(rsi_value):
    if pd.isna(rsi_value): return np.nan
    if rsi_value >= RSI_THRESHOLDS['summer']: return '여름'
    if rsi_value >= 50: return '봄'
    if rsi_value >= RSI_THRESHOLDS['winter']: return '가을'
    return '겨울'

def run_dynamic_strategy(sp500_df, rsi_series, start_date, end_date, 
                        initial_capital=10000000, strategy_rules=None, special_strategy=None):
    """동적 RSI 기반 로테이션 전략의 완전한 로직"""
    
    if strategy_rules is None:
        return None
    
    monthly_sp500 = sp500_df.resample('M').last()
    monthly_rsi = rsi_series.resample('M').last()
    
    monthly_sp500 = monthly_sp500[(monthly_sp500.index >= start_date) & (monthly_sp500.index <= end_date)]
    monthly_rsi = monthly_rsi[(monthly_rsi.index >= start_date) & (monthly_rsi.index <= end_date)]
    
    portfolio_values = []
    monthly_returns_log = []
    cash = initial_capital
    current_style = None
    current_shares = 0
    
    season_history = []
    in_sideways_mode = False
    
    for i, date in enumerate(monthly_sp500.index):
        if date in monthly_rsi.index:
            current_rsi = monthly_rsi.loc[date]
            current_season = classify_market_season(current_rsi)
        else:
            current_season = monthly_returns_log[-1]['season'] if i > 0 and len(monthly_returns_log) > 0 else '봄'
            current_rsi = 50.0
        
        if pd.notna(current_season):
            season_history.append(current_season)
            if len(season_history) > 5:
                season_history.pop(0)
        
        if special_strategy:
            if not in_sideways_mode:
                if len(season_history) >= 2:
                    prev_season = season_history[-2]
                    if ((prev_season == '가을' and current_season == '봄') or (prev_season == '봄' and current_season == '가을')):
                        in_sideways_mode = True
            elif in_sideways_mode and current_season in ['여름', '겨울']:
                in_sideways_mode = False
        
        if in_sideways_mode and special_strategy and current_season in special_strategy:
            target_style = special_strategy[current_season]
        else:
            target_style = strategy_rules.get(current_season, 'Quality')
        
        style_mapping = {
            'Momentum': 'S&P500 Momentum', 'Quality': 'S&P500 Quality', 'Low Vol': 'S&P500 Low Volatiltiy Index',
            'Growth': 'S&P500 Growth', 'Value': 'S&P500 Value', 'Dividend': 'S&P500 Div Aristocrt TR Index', 'S&P500': 'S&P500'
        }
        actual_style = style_mapping.get(target_style, 'S&P500 Quality')

        if actual_style not in monthly_sp500.columns:
            actual_style = monthly_sp500.columns[0]

        portfolio_value = current_shares * monthly_sp500.loc[date, current_style] if current_style and current_shares > 0 else cash
        
        if actual_style != current_style:
            if current_style and current_shares > 0:
                sell_price = monthly_sp500.loc[date, current_style]
                cash = current_shares * sell_price
            
            buy_price = monthly_sp500.loc[date, actual_style]
            if buy_price > 0:
                current_shares = cash / buy_price
                cash = 0
            current_style = actual_style
            portfolio_value = current_shares * monthly_sp500.loc[date, current_style]

        portfolio_values.append(portfolio_value)
        monthly_returns_log.append({'season': current_season})

    if not portfolio_values:
        return {"total_return": 0, "cagr": 0, "portfolio_series": pd.Series()}

    portfolio_series = pd.Series(portfolio_values, index=monthly_sp500.index)
    final_value = portfolio_series.iloc[-1]
    total_return = (final_value / initial_capital) - 1
    years = (monthly_sp500.index[-1] - monthly_sp500.index[0]).days / 365.25
    cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    return {"total_return": total_return, "cagr": cagr, "portfolio_series": portfolio_series}


# --- API 엔드포인트 ---
@app.post("/backtest")
async def backtest(request: BacktestRequest) -> Dict[str, Any]:
    """백테스팅을 실행하고 결과를 반환하는 API"""
    
    start_date = pd.Timestamp('1999-01-01')
    end_date = pd.Timestamp('2025-06-30')
    
    # 커스텀 전략 실행
    custom_result = run_dynamic_strategy(
        sp500_df, rsi_series, start_date, end_date,
        strategy_rules=request.strategy_rules.dict(),
        special_strategy=request.special_strategy.dict() if request.special_strategy else None
    )

    # 기본 보유 전략들과 비교
    predefined_strategies = {f'{s} 보유': {sea: s for sea in ['여름', '봄', '가을', '겨울']} for s in ['Momentum', 'Quality', 'Growth', 'Value', 'Low Vol', 'Dividend']}
    
    all_results = {request.strategy_name: custom_result}
    for name, rules in predefined_strategies.items():
        all_results[name] = run_dynamic_strategy(sp500_df, rsi_series, start_date, end_date, strategy_rules=rules)

    # 결과 차트 생성
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, result in all_results.items():
        ax.plot(result['portfolio_series'].index, result['portfolio_series'].values, label=name)
    
    ax.set_title('전략별 포트폴리오 가치 변화', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.5)
    
    # 차트를 이미지 데이터로 변환
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # 프론트엔드로 보낼 최종 결과 데이터 구성
    summary = {
        name: {"total_return": f"{res['total_return']:.2%}", "cagr": f"{res['cagr']:.2%}"}
        for name, res in all_results.items()
    }

    return {
        "summary": summary,
        "chart": "data:image/png;base64," + chart_base64
    }

# CORS (Cross-Origin Resource Sharing) 설정 - 프론트엔드와 통신하기 위함
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 운영 시에는 프론트엔드 주소만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
