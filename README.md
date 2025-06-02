Lorenz Attractor Simulation

 Lorenz 어트랙터(카오스 이론 대표 모델)를 수치 통합하고
Lyapunov 지수·프랙탈 차원을 정량화하며, 민감도 분석·시각화·애니메이션·GUI

🚀 주요 기능

수치 통합 & 물리 모델– simulation/solver.py: RK4 기반 ODE 통합기 & Lorenz 시스템 우변– simulation/lorenz.py: 3D 궤적 시뮬레이션 및 Matplotlib 시각화

카오스 정량화– simulation/lyapunov.py: Wolf 방법으로 최대 Lyapunov 지수 추정– simulation/fractal_dimension.py: 박스-카운팅 방식 프랙탈 차원 계산

민감도 & 파라미터 스윕– analysis/sensitivity.py: 초기 조건 민감도 분석– analysis/parameter_sweep.py: σ·ρ·초기조건 스윕 → Lyapunov·Fractal 결과 CSV 저장

시각화 & 애니메이션– visualization/plot3d.py: Matplotlib 3D 정적 플롯 (시간 컬러 매핑)– visualization/animation.py: Matplotlib 애니메이션(GIF/MP4)– visualization/plot_stats.py: 히트맵·박스플롯 등 통계 시각화– visualization/interactive_gui.py: Streamlit 기반 인터랙티브 시뮬레이터

통합 CLI– main.py: simulate, animate, lyapunov, fractal, sensitivity, gui 커맨드를 지원


# requirements.txt
numpy>=1.20
scipy>=1.6
matplotlib>=3.4
plotly>=5.0
pandas>=1.3
streamlit>=1.10
imageio-ffmpeg>=0.4

📁 디렉터리 구조

lorenz_project/
├── simulation/
│   ├── solver.py
│   ├── lorenz.py
│   ├── lyapunov.py
│   └── fractal_dimension.py
│
├── analysis/
│   ├── sensitivity.py
│   ├── parameter_sweep.py
│   └── summarize.py
│
├── visualization/
│   ├── plot3d.py
│   ├── animation.py
│   ├── plot_stats.py
│   └── interactive_gui.py
│
├── main.py
├── README.md
└── requirements.txt

💻 사용 예시

1. CLI 실행

# (1) 기본 시뮬레이션 + 3D 플롯
python main.py simulate \
  --t0 0 --t1 50 --dt 0.01 \
  --x0 1 --y0 1 --z0 1 \
  --sigma 10 --rho 28 --beta 2.6667 \
  --save lorenz_plot.png

# (2) 3D 애니메이션 생성
python main.py animate \
  --frames 500 --interval 20 \
  --save lorenz_anim.mp4

# (3) Lyapunov 지수 계산
python main.py lyapunov \
  --t 100 --dt 0.01 --d0 1e-8 \
  --interval 0.1 --transient 5

# (4) 프랙탈 차원 추정
python main.py fractal \
  --scales 12 --save fractal_result.txt

# (5) 초기조건 민감도 분석
python main.py sensitivity \
  --delta 1e-6 --save sensitivity.png

# (6) Streamlit GUI 실행
streamlit run visualization/interactive_gui.py
# 또는
python main.py gui

2. 파라미터 스윕 & 통계 분석

# σ, ρ, x0,y0,z0 조합별 Lyapunov·Fractal 계산 → CSV 저장
python analysis/parameter_sweep.py

# 결과 요약 & 시각화
python analysis/summarize.py
python visualization/plot_stats.py

⚙️ 커스터마이징

파라미터: σ, ρ, β, dt, T₁ 등은 CLI/GUI 옵션 혹은 config.py에서 조정

초기조건: analysis/parameter_sweep.py 내 INITIAL_CONDITIONS 리스트 수정

시각화: plot3d.py, animation.py 파라미터(컬러맵·뷰각) 변경 가능

📖 물리적 해석

Lyapunov 지수: ρ 증가 시 양의 지수가 커져 “예측 불가능성” 강화

프랙탈 차원: 약 2.06 → 궤적이 2D 면보다 복잡하지만, 3D를 완전히 채우진 않음

민감도 분석: 초기조건 미소 변화에 따른 궤적 분기(“나비 효과”) 정량화