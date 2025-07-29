import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def improved_hmm_preprocessing(df, variables):
    """강화된 HMM 전처리 함수"""
    
    print("=== 강화된 HMM 전처리 시작 ===")
    df_processed = df.copy()
    
    # 1. 이상치 처리 (Winsorization) - 상위/하위 1% 제거
    print("1. 이상치 처리 (Winsorization):")
    for col in variables:
        if col in df_processed.columns:
            q01 = df_processed[col].quantile(0.01)
            q99 = df_processed[col].quantile(0.99)
            original_range = df_processed[col].max() - df_processed[col].min()
            
            df_processed[col] = df_processed[col].clip(lower=q01, upper=q99)
            
            new_range = df_processed[col].max() - df_processed[col].min()
            reduction = (1 - new_range/original_range) * 100
            print(f"   {col}: {reduction:.1f}% 범위 축소")
    
    # 2. 결측치 처리
    print("2. 결측치 처리:")
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=variables).reset_index(drop=True)
    final_rows = len(df_processed)
    print(f"   {initial_rows} → {final_rows} 행 ({final_rows/initial_rows*100:.1f}% 유지)")
    
    # 3. RobustScaler 적용
    print("3. RobustScaler 적용:")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_processed[variables])
    
    # 스케일링 효과 확인
    original_var_ratio = df_processed[variables].var().max() / df_processed[variables].var().min()
    scaled_var_ratio = X_scaled.var(axis=0).max() / X_scaled.var(axis=0).min()
    
    print(f"   원본 분산 비율: {original_var_ratio:.1f}")
    print(f"   스케일링 후 분산 비율: {scaled_var_ratio:.2f}")
    print(f"   분산 균일화 개선도: {original_var_ratio/scaled_var_ratio:.1f}배")
    
    return df_processed, X_scaled, scaler

def ensemble_hmm_training(X_scaled, n_states=4, n_trials=15):
    """앙상블 방식 HMM 학습"""
    
    print("\n=== 앙상블 HMM 학습 ===")
    
    # 다양한 설정으로 학습
    configs = [
        {'covariance_type': 'diag', 'description': '대각 공분산'},
        {'covariance_type': 'spherical', 'description': '구형 공분산'},
        {'covariance_type': 'tied', 'description': '연결 공분산'}
    ]
    
    best_model = None
    best_states = None
    best_score = 0
    best_config = None
    
    print("학습 진행 상황:")
    
    for i, config in enumerate(configs):
        print(f"  {config['description']} 테스트 중...")
        
        for trial in range(n_trials // len(configs)):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=config['covariance_type'],
                n_iter=1000,
                tol=1e-4,
                random_state=42 + trial + i*10
            )
            
            try:
                model.fit(X_scaled)
                states = model.predict(X_scaled)
                
                # 균형도 평가
                score = evaluate_state_balance(states, n_states)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_states = states
                    best_config = config['description']
                    
            except Exception as e:
                continue
    
    print(f"\n최적 모델 선택 완료:")
    print(f"  설정: {best_config}")
    print(f"  균형도 점수: {best_score:.3f}")
    
    return best_model, best_states, best_score

def evaluate_state_balance(states, n_states):
    """상태 균형도 평가 함수"""
    
    # 상태별 빈도 계산
    state_counts = np.bincount(states, minlength=n_states)
    state_probs = state_counts / len(states)
    
    # 엔트로피 기반 균형도 (0~1, 1이 완전 균형)
    entropy = -np.sum(state_probs * np.log(state_probs + 1e-10))
    max_entropy = np.log(n_states)
    balance_score = entropy / max_entropy
    
    # 최소 빈도 제약 (각 상태가 최소 5% 이상이어야 함)
    min_freq = np.min(state_probs)
    if min_freq < 0.05:
        penalty = min_freq / 0.05  # 5% 미만이면 페널티
        balance_score *= penalty
    
    return balance_score

def analyze_hmm_results(df_processed, states, variables, model):
    """HMM 결과 분석"""
    
    print("\n=== HMM 결과 분석 ===")
    
    n_states = len(np.unique(states))
    
    # 1. 상태별 기본 통계
    print("1. 상태별 분포:")
    state_counts = np.bincount(states, minlength=n_states)
    for i in range(n_states):
        percentage = state_counts[i] / len(states) * 100
        print(f"   State {i}: {state_counts[i]}개 ({percentage:.1f}%)")
    
    # 2. 상태별 변수 특성
    print("\n2. 상태별 주요 특성:")
    df_analysis = df_processed.copy()
    df_analysis['state'] = states
    
    state_characteristics = {}
    
    for state in range(n_states):
        mask = states == state
        if np.sum(mask) > 0:
            state_data = df_analysis.loc[mask, variables]
            means = state_data.mean()
            
            # 특성 분석 (표준화된 값 기준)
            characteristics = []
            
            if 'S&P500_rate' in means.index:
                if means['S&P500_rate'] > 0.01:
                    characteristics.append("주식 상승")
                elif means['S&P500_rate'] < -0.01:
                    characteristics.append("주식 하락")
            
            if 'VIX_SPX' in means.index:
                if means['VIX_SPX'] > 0.05:
                    characteristics.append("변동성 높음")
                elif means['VIX_SPX'] < -0.05:
                    characteristics.append("변동성 낮음")
            
            if 'CreditSpread' in means.index:
                if means['CreditSpread'] > 5:
                    characteristics.append("신용위험 증가")
                elif means['CreditSpread'] < -5:
                    characteristics.append("신용위험 감소")
            
            if 'UnempRate' in means.index:
                if means['UnempRate'] > 0.2:
                    characteristics.append("실업률 상승")
                elif means['UnempRate'] < -0.2:
                    characteristics.append("실업률 하락")
            
            state_characteristics[state] = characteristics
            print(f"   State {state}: {', '.join(characteristics) if characteristics else '중립적'}")
    
    # 3. 상태 지속성 분석
    print("\n3. 상태 지속성:")
    transitions = calculate_state_persistence(states)
    for i in range(n_states):
        persistence = transitions[i, i] if transitions[i].sum() > 0 else 0
        print(f"   State {i}: {persistence:.1%} 지속 확률")
    
    return state_characteristics

def calculate_state_persistence(states):
    """상태 전이 확률 계산"""
    
    n_states = len(np.unique(states))
    transitions = np.zeros((n_states, n_states))
    
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transitions[current_state, next_state] += 1
    
    # 행별 정규화 (각 상태에서 다른 상태로의 전이 확률)
    for i in range(n_states):
        row_sum = transitions[i].sum()
        if row_sum > 0:
            transitions[i] = transitions[i] / row_sum
    
    return transitions

def plot_improved_hmm_results(df_processed, states, variables):
    """개선된 HMM 결과 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('강화된 전처리 HMM 결과', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange']
    n_states = len(np.unique(states))
    
    # 1. 상태별 빈도
    ax1 = axes[0, 0]
    state_counts = np.bincount(states, minlength=n_states)
    bars = ax1.bar(range(n_states), state_counts, color=colors[:n_states], alpha=0.7)
    ax1.set_title('상태별 빈도')
    ax1.set_xlabel('상태')
    ax1.set_ylabel('빈도')
    ax1.set_xticks(range(n_states))
    ax1.set_xticklabels([f'State {i}' for i in range(n_states)])
    
    # 빈도 표시
    for bar, count in zip(bars, state_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom')
    
    # 2. S&P500 기준 상태 분포 (날짜 데이터가 있는 경우)
    ax2 = axes[0, 1]
    if 'date' in df_processed.columns and 'S&P500' in df_processed.columns:
        dates = df_processed['date']
        sp500 = df_processed['S&P500']
        
        ax2.plot(dates, sp500, color='gray', alpha=0.5, linewidth=1)
        
        for state in range(n_states):
            mask = states == state
            if np.any(mask):
                ax2.scatter(dates[mask], sp500.values[mask],
                           c=colors[state], s=15, alpha=0.8, label=f'State {state}')
        
        ax2.set_title('S&P500 지수별 상태 분포')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'S&P500 데이터 없음', ha='center', va='center', 
                transform=ax2.transAxes)
        ax2.set_title('S&P500 지수별 상태 분포')
    
    # 3. 상태 전이 확률
    ax3 = axes[0, 2]
    transitions = calculate_state_persistence(states)
    
    im = ax3.imshow(transitions, cmap='Blues', alpha=0.8)
    ax3.set_title('상태 전이 확률')
    ax3.set_xlabel('다음 상태')
    ax3.set_ylabel('현재 상태')
    
    for i in range(n_states):
        for j in range(n_states):
            ax3.text(j, i, f'{transitions[i,j]:.2f}',
                    ha='center', va='center', fontsize=10)
    
    ax3.set_xticks(range(n_states))
    ax3.set_yticks(range(n_states))
    
    # 4. 균형도 점수
    ax4 = axes[1, 0]
    balance_score = evaluate_state_balance(states, n_states)
    
    ax4.bar(['균형도'], [balance_score], color='lightblue', alpha=0.7)
    ax4.set_title('상태 균형도')
    ax4.set_ylabel('균형도 점수')
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='완전균형')
    ax4.text(0, balance_score + 0.05, f'{balance_score:.3f}', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # 5. 시계열 상태 변화
    ax5 = axes[1, 1]
    ax5.plot(range(len(states)), states, marker='o', markersize=2, alpha=0.7)
    ax5.set_title('시간별 상태 변화')
    ax5.set_xlabel('시간 순서')
    ax5.set_ylabel('상태')
    ax5.set_yticks(range(n_states))
    ax5.grid(True, alpha=0.3)
    
    # 6. 상태별 변수 히트맵
    ax6 = axes[1, 2]
    
    # 상태별 평균 계산
    state_means = []
    for state in range(n_states):
        mask = states == state
        if np.any(mask):
            means = df_processed.loc[mask, variables].mean()
        else:
            means = pd.Series(0, index=variables)
        state_means.append(means.values)
    
    state_means_array = np.array(state_means)
    
    # 정규화 (각 변수별로)
    for j in range(len(variables)):
        col_data = state_means_array[:, j]
        if col_data.std() > 0:
            state_means_array[:, j] = (col_data - col_data.mean()) / col_data.std()
    
    im6 = ax6.imshow(state_means_array, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
    ax6.set_title('상태별 변수 특성 (정규화)')
    ax6.set_xlabel('변수')
    ax6.set_ylabel('상태')
    ax6.set_xticks(range(len(variables)))
    ax6.set_xticklabels(variables, rotation=45, ha='right')
    ax6.set_yticks(range(n_states))
    ax6.set_yticklabels([f'State {i}' for i in range(n_states)])
    
    plt.colorbar(im6, ax=ax6)
    plt.tight_layout()
    
    return fig

# 메인 실행 함수
def run_improved_hmm(df, variables, n_states=4):
    """개선된 HMM 전체 실행"""
    
    print("강화된 전처리 HMM 분석 시작")
    print("=" * 50)
    
    # 1. 강화된 전처리
    df_processed, X_scaled, scaler = improved_hmm_preprocessing(df, variables)
    
    # 2. 앙상블 HMM 학습
    model, states, balance_score = ensemble_hmm_training(X_scaled, n_states)
    
    # 3. 결과 분석
    characteristics = analyze_hmm_results(df_processed, states, variables, model)
    
    # 4. 시각화
    fig = plot_improved_hmm_results(df_processed, states, variables)
    plt.show()
    
    # 5. 결과 반환
    df_processed['state'] = states
    
    results = {
        'model': model,
        'states': states,
        'balance_score': balance_score,
        'state_characteristics': characteristics,
        'processed_data': df_processed,
        'scaler': scaler
    }
    
    print(f"\n✅ 분석 완료! 균형도 점수: {balance_score:.3f}")
    
    return results