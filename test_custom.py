# 커스텀 전략 테스트를 위한 자동 입력 스크립트
import subprocess
import sys

# 모드 2 선택, 각 계절별로 1,2,3,4 선택, 전략명 '테스트전략', 확인 'y'
inputs = "2\n1\n2\n3\n4\n테스트전략\ny\n"

try:
    process = subprocess.run(
        [sys.executable, "Enhanced_Backtesting_v1.0.py"],
        input=inputs,
        text=True,
        capture_output=True,
        timeout=60
    )
    
    print("STDOUT:")
    print(process.stdout)
    
    if process.stderr:
        print("STDERR:")
        print(process.stderr)
        
    print(f"Return code: {process.returncode}")
    
except subprocess.TimeoutExpired:
    print("프로세스가 타임아웃되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")