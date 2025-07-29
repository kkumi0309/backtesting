"""
HMM 시장 국면 탐지 분석 - 원클릭 실행 스크립트

사용법:
1. 이 파일을 data_20250726.csv, sp500_data.csv와 같은 폴더에 저장
2. Python으로 실행
3. 자동으로 모든 분석 수행 및 결과 출력

필요한 라이브러리:
pip install pandas numpy matplotlib seaborn scikit-learn hmmlearn openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

def simple_hmm_analysis():
    """
    간단한 HMM 분석 (핵심 기능만)
    """
    print("🚀 HMM 시장 국면 분석 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    try:
        # 거시경제 데이터
        macro = pd.read_csv('data_20250726.csv')
        macro.columns = [col if col else 'Date' for col in macro.columns]  # 빈 컬럼명 처리
        macro['Date'] = pd.to_datetime(macro['Date'], errors='coerce')
        macro = macro.set_index('Date').dropna()
        print(f"✅ 거시경제 데이터: {macro.shape}")
        
        # 스타일 데이터
        try:
            style = pd.read_csv('sp500_data.csv')
            style['날짜'] = pd.to_datetime(style['날짜'], errors='coerce')
            style = style.set_index('날짜').dropna()
            print(f"✅ 스타일 데이터: {style.shape}")
        except:
            style = None
            print("⚠️ 스타일 데이터 없음 (거시데이터만 사용)")
            
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None
    
    # 2. 피처 생성
    print("\n🔧 피처 생성 중...")
    features = pd.DataFrame(index=macro.index)
    
    # 거시경제 지표 처리
    for col in macro.columns:
        series = macro[col]
        if series.dtype == 'object':
            # 문자열 데이터 정제
            series = series.astype(str).str.replace('%', '').str.replace(',', '')
            series = pd.to_numeric(series, errors='coerce')
        features[col.replace(' ', '_').replace('&', '')] = series
    
    # S&P 500 수익률 계산
    if 'SP500' in features.columns:
        features['SP500_Return'] = features['SP500'].pct_change() * 100
    
    # 스타일 수익률 추가 (있는 경우)
    if style is not None:
        for col in style.columns:
            if style[col].dtype in ['float64', 'int64']:
                returns = style[col].pct_change() * 100
                clean_name = col.replace('S&P500 ', '').replace(' ', '_')
                features[f'Style_{clean_name}'] = returns
    
    # 결측치 제거 및 표준화
    features = features.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    
    print(f"   최종 피처: {features.shape}")
    print(f"   기간: {features.index.min()} ~ {features.index.max()}")
    
    # 3. HMM 최적화
    print("\n🤖 HMM 모델 학습 중...")
    best_model = None
    best_bic = np.inf
    best_n = 3
    
    for n_states in range(2, 6):
        try:
            model = hmm.GaussianHMM(n_components=n_states, random_state=42, n_iter=100)
            model.fit(X)
            
            ll = model.score(X)
            n_params = n_states * (n_states + 2 * X.shape[1])
            bic = -2 * ll + np.log(len(X)) * n_params
            
            print(f"   {n_states}개 상태: BIC = {bic:.2f}")
            
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n_states
        except:
            continue
    
    if best_model is None:
        print("❌ 모델 학습 실패")
        return None
    
    print(f"✅ 최적 모델: {best_n}개 상태")
    
    # 4. 국면 분석
    states = best_model.predict(X)
    
    print(f"\n📊 국면 분석 결과")
    print("=" * 50)
    
    regime_info = {}
    for state in range(best_n):
        mask = states == state
        if mask.sum() == 0:
            continue
            
        state_data = features[mask]
        duration = len(state_data)
        frequency = duration / len(features) * 100
        
        # 주요 특징
        mean_vals = state_data.mean()
        sp500_ret = mean_vals.get('SP500_Return', 0)
        
        # 간단한 의미 부여
        if sp500_ret > 0.5:
            meaning = "강세장"
        elif sp500_ret > 0:
            meaning = "회복기"
        elif sp500_ret < -0.5:
            meaning = "약세장"
        else:
            meaning = "조정기"
        
        regime_info[state] = {
            'meaning': meaning,
            'duration': duration,
            'frequency': frequency
        }
        
        print(f"State {state}: {meaning}")
        print(f"  지속: {duration}개월 ({frequency:.1f}%)")
        print(f"  수익률: {sp500_ret:.3f}")
    
    # 5. 시각화
    print(f"\n📈 결과 시각화")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HMM 시장 국면 분석 결과', fontsize=14, fontweight='bold')
    
    # 국면 시계열
    axes[0,0].plot(features.index, states, 'o-', markersize=3)
    axes[0,0].set_title('시장 국면 변화')
    axes[0,0].set_ylabel('State')
    axes[0,0].grid(True, alpha=0.3)
    
    # 국면별 지속기간
    unique_states, counts = np.unique(states, return_counts=True)
    state_labels = [f"S{s}\n{regime_info[s]['meaning']}" for s in unique_states if s in regime_info]
    axes[0,1].bar(range(len(unique_states)), counts)
    axes[0,1].set_xticks(range(len(unique_states)))
    axes[0,1].set_xticklabels(state_labels, rotation=45)
    axes[0,1].set_title('국면별 지속기간')
    
    # S&P 500 수익률과 국면
    if 'SP500_Return' in features.columns:
        colors = plt.cm.Set3(states / max(states))
        axes[1,0].scatter(features.index, features['SP500_Return'], c=colors, alpha=0.7)
        axes[1,0].set_title('S&P 500 수익률과 국면')
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1,0].grid(True, alpha=0.3)
    
    # 전환 확률 행렬
    sns.heatmap(best_model.transmat_, annot=True, ax=axes[1,1], fmt='.3f')
    axes[1,1].set_title('국면 전환 확률')
    
    plt.tight_layout()
    plt.show()
    
    # 6. 현재 상황 및 투자 제안
    current_state = states[-1]
    current_meaning = regime_info[current_state]['meaning']
    
    print(f"\n💼 투자 제안")
    print("=" * 50)
    print(f"현재 시장 국면: {current_meaning} (State {current_state})")
    
    # 간단한 투자 제안
    investment_advice = {
        "강세장": "주식 70-80%, 성장주 중심 투자",
        "회복기": "주식 60-70%, 가치주 발굴 기회",
        "약세장": "주식 20-30%, 현금/채권 비중 확대",
        "조정기": "주식 50%, 변동성 관리 중요"
    }
    
    advice = investment_advice.get(current_meaning, "균형 포트폴리오 유지")
    print(f"권장 전략: {advice}")
    
    # 7. 결과 저장
    print(f"\n💾 결과 저장")
    
    # CSV 저장
    result_df = pd.DataFrame({
        'Date': features.index,
        'Market_Regime': states,
        'Economic_Meaning': [regime_info[s]['meaning'] for s in states]
    })
    
    if 'SP500_Return' in features.columns:
        result_df['SP500_Return'] = features['SP500_Return'].values
    
    result_df.to_csv('market_regime_simple_results.csv', index=False)
    print("✅ 결과 저장: market_regime_simple_results.csv")
    
    return {
        'model': best_model,
        'states': states,
        'features': features,
        'regime_info': regime_info,
        'current_state': current_state,
        'current_meaning': current_meaning
    }

def print_detailed_results(results):
    """상세 결과 출력"""
    if not results:
        return
    
    print(f"\n📋 상세 분석 결과")
    print("=" * 50)
    
    regime_info = results['regime_info']
    features = results['features']
    
    for state, info in regime_info.items():
        print(f"\n🔍 State {state}: {info['meaning']}")
        print(f"   지속기간: {info['duration']}개월")
        print(f"   전체 비중: {info['frequency']:.1f}%")
        
        # 해당 국면의 기간들
        state_mask = results['states'] == state
        state_periods = features.index[state_mask]
        
        if len(state_periods) > 0:
            print(f"   주요 기간: {state_periods[0]} ~ {state_periods[-1]}")

if __name__ == "__main__":
    print("🎯 HMM 시장 국면 탐지 분석 - 간단 버전")
    print("필요 파일: data_20250726.csv, sp500_data.csv")
    print("=" * 60)
    
    # 분석 실행
    results = simple_hmm_analysis()
    
    if results:
        print("\n🎉 분석 성공!")
        print("\n📊 생성된 결과:")
        print("• 시각화: 4개 차트")
        print("• CSV 파일: market_regime_simple_results.csv")
        print("• 현재 시장 상황 및 투자 제안")
        
        # 상세 결과 출력
        print_detailed_results(results)
        
        print(f"\n💡 활용 방법:")
        print("• CSV 파일로 과거 국면 패턴 분석")
        print("• 현재 국면에 맞는 투자 전략 적용")
        print("• 정기적 업데이트로 국면 변화 모니터링")
        
    else:
        print("❌ 분석 실패. 데이터 파일을 확인해주세요.")
    
    print(f"\n🔧 문제 해결:")
    print("• 파일명 확인: data_20250726.csv, sp500_data.csv")
    print("• 같은 폴더에 파일 위치")
    print("• 라이브러리 설치: pip install pandas numpy matplotlib seaborn scikit-learn hmmlearn")