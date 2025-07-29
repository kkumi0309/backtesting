# 횡보 구간 전략 테스트를 위한 자동 입력 스크립트
import subprocess
import sys

# 기본 전략: 4(Quality), 3(Momentum), 5(Low Vol), 5(Low Vol)
# 특별 조건 사용: y
# 횡보 구간 봄: 3(Momentum)
# 횡보 구간 가을: 4(Quality)  
# 전략명: 횡보전략
# 확인: y
inputs = "4\n3\n5\n5\ny\n3\n4\n횡보전략\ny\n"

try:
    process = subprocess.run(
        [sys.executable, "Enhanced_Backtesting_v1.0.py"],
        input=inputs,
        text=True,
        capture_output=True,
        timeout=120  # 시간 늘림
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