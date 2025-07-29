import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// --- 타입 정의 ---
interface Strategy {
  [key: string]: string;
}

interface ResultSummary {
  [strategyName: string]: {
    total_return: string;
    cagr: string;
  };
}

interface BacktestResponse {
  summary: ResultSummary;
  chart: string; // base64 encoded image string
}

const initialStrategy: Strategy = {
  봄: 'Momentum',
  여름: 'Quality',
  가을: 'Low Vol',
  겨울: 'Value',
};

const initialSpecialStrategy: Strategy = {
  봄: 'Quality',
  가을: 'Value',
}

const styleOptions = ['Momentum', 'Quality', 'Growth', 'Value', 'Low Vol', 'Dividend'];

function App() {
  const [strategy, setStrategy] = useState<Strategy>(initialStrategy);
  const [useSpecial, setUseSpecial] = useState(false);
  const [specialStrategy, setSpecialStrategy] = useState<Strategy>(initialSpecialStrategy);
  const [strategyName, setStrategyName] = useState('사용자 전략');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState('');

  const handleStrategyChange = (season: string, value: string, isSpecial: boolean) => {
    if (isSpecial) {
      setSpecialStrategy(prev => ({ ...prev, [season]: value }));
    } else {
      setStrategy(prev => ({ ...prev, [season]: value }));
    }
  };

  const handleRunBacktest = async () => {
    setLoading(true);
    setError('');
    setResults(null);

    const requestBody = {
      strategy_name: strategyName,
      strategy_rules: strategy,
      special_strategy: useSpecial ? specialStrategy : null,
    };

    try {
      const response = await axios.post<BacktestResponse>('http://127.0.0.1:8000/backtest', requestBody);
      setResults(response.data);
    } catch (err) {
      setError('백테스트 실행 중 오류가 발생했습니다. 백엔드 서버가 실행 중인지 확인하세요.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <header className="text-center mb-4">
        <h1>동적 자산배분 백테스팅</h1>
        <p className="lead">RSI 계절 국면을 이용한 투자 전략을 테스트합니다.</p>
      </header>

      <div className="card shadow-sm mb-5">
        <div className="card-header">
          <h3>전략 설정</h3>
        </div>
        <div className="card-body">
          <div className="mb-3">
            <label htmlFor="strategyName" className="form-label">전략 이름</label>
            <input 
              type="text"
              className="form-control"
              id="strategyName"
              value={strategyName}
              onChange={(e) => setStrategyName(e.target.value)}
            />
          </div>
          <h5 className="mt-4">기본 전략</h5>
          <div className="row">
            {Object.keys(initialStrategy).map(season => (
              <div className="col-md-3 col-6 mb-3" key={season}>
                <label htmlFor={season} className="form-label fw-bold">{season}</label>
                <select 
                  id={season} 
                  className="form-select"
                  value={strategy[season]}
                  onChange={(e) => handleStrategyChange(season, e.target.value, false)}
                >
                  {styleOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </div>
            ))}
          </div>

          <hr className="my-4" />

          <div className="form-check form-switch mb-3">
            <input 
              className="form-check-input" 
              type="checkbox" 
              role="switch" 
              id="useSpecialStrategy"
              checked={useSpecial}
              onChange={(e) => setUseSpecial(e.target.checked)}
            />
            <label className="form-check-label" htmlFor="useSpecialStrategy"><h5>횡보 구간 특별 조건 사용</h5></label>
          </div>

          {useSpecial && (
            <div id="special-strategy-options">
              <p className="text-muted">봄-가을 횡보 구간에서 사용할 전략을 별도로 설정합니다.</p>
              <div className="row">
                {Object.keys(initialSpecialStrategy).map(season => (
                  <div className="col-md-3 col-6 mb-3" key={`special-${season}`}>
                    <label htmlFor={`special-${season}`} className="form-label fw-bold text-danger">횡보 {season}</label>
                    <select 
                      id={`special-${season}`} 
                      className="form-select border-danger"
                      value={specialStrategy[season]}
                      onChange={(e) => handleStrategyChange(season, e.target.value, true)}
                    >
                      {styleOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="d-grid mt-4">
            <button 
              className="btn btn-primary btn-lg" 
              onClick={handleRunBacktest} 
              disabled={loading}
            >
              {loading ? '백테스팅 실행 중...' : '백테스팅 실행'}
            </button>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {results && (
        <div className="card shadow-sm">
          <div className="card-header">
            <h3>백테스팅 결과</h3>
          </div>
          <div className="card-body">
            <h4 className="mb-3">성과 요약</h4>
            <table className="table table-striped table-hover">
              <thead className="table-dark">
                <tr>
                  <th>전략</th>
                  <th>총수익률</th>
                  <th>연평균수익률(CAGR)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.summary).sort((a, b) => a[0] === strategyName ? -1 : b[0] === strategyName ? 1 : 0).map(([name, metrics]) => (
                  <tr key={name} className={name === strategyName ? 'table-danger' : ''}>
                    <td className='fw-bold'>{name}</td>
                    <td>{metrics.total_return}</td>
                    <td>{metrics.cagr}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h4 className="mt-5 mb-3">포트폴리오 가치 변화</h4>
            <div className="text-center">
              <img src={results.chart} alt="Backtest Chart" className="img-fluid border rounded"/>
            </div>
          </div>
        </div>
      )}

      <footer className="text-center text-muted mt-5 mb-3">
        <p>백테스팅 기간: 1999-01-01 ~ 2025-06-30</p>
      </footer>
    </div>
  );
}

export default App;
