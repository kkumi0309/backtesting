import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// 백엔드 응답 데이터 형식을 정의합니다.
interface BacktestResult {
  cagr: number;
  mdd: number;
  sharpe_ratio: number;
}

interface BacktestResponse {
  message: string;
  results: BacktestResult;
  plot_html: string;
}

function App() {
  // 상태 변수들을 정의합니다.
  const [strategy, setStrategy] = useState('sma_cross');
  const [ticker, setTicker] = useState('SPY');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState('2023-12-31');
  const [shortWindow, setShortWindow] = useState(20);
  const [longWindow, setLongWindow] = useState(60);
  
  const [results, setResults] = useState<BacktestResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 폼 제출 핸들러
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setResults(null);

    const requestBody = {
      strategy,
      ticker,
      start_date: startDate,
      end_date: endDate,
      parameters: {
        short_window: shortWindow,
        long_window: longWindow,
      },
    };

    try {
      // 백엔드 API를 호출합니다. 이 URL이 가장 중요한 변경 사항입니다.
      const response = await axios.post<BacktestResponse>('https://backtesting-backend.onrender.com/backtest', requestBody);
      setResults(response.data);
    } catch (err) {
      setError('백테스트 실행 중 오류가 발생했습니다. 잠시 후 다시 시도하거나 백엔드 서버 상태를 확인하세요.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Dynamic Backtester</h1>
      </header>
      <main>
        <form onSubmit={handleSubmit} className="backtest-form">
          <h2>전략 설정</h2>
          {/* 입력 필드들 */}
          <div className="form-grid">
            <div>
              <label>전략:</label>
              <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                <option value="sma_cross">SMA Crossover</option>
              </select>
            </div>
            <div>
              <label>티커:</label>
              <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
            </div>
            <div>
              <label>시작일:</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            </div>
            <div>
              <label>종료일:</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </div>
            <div>
              <label>단기 이동평균:</label>
              <input type="number" value={shortWindow} onChange={(e) => setShortWindow(parseInt(e.target.value))} />
            </div>
            <div>
              <label>장기 이동평균:</label>
              <input type="number" value={longWindow} onChange={(e) => setLongWindow(parseInt(e.target.value))} />
            </div>
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? '백테스트 실행 중...' : '백테스트 실행'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {results && (
          <div className="results-container">
            <h2>백테스트 결과</h2>
            <div className="metrics">
              <p><strong>CAGR:</strong> {(results.results.cagr * 100).toFixed(2)}%</p>
              <p><strong>MDD:</strong> {(results.results.mdd * 100).toFixed(2)}%</p>
              <p><strong>샤프 지수:</strong> {results.results.sharpe_ratio.toFixed(2)}</p>
            </div>
            <div className="plot" dangerouslySetInnerHTML={{ __html: results.plot_html }} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
