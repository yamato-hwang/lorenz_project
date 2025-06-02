"""
visualization/dynamics_comparison.py

Lorenz 시스템의 다양한 동역학 상태(원점 수렴, 정적 평형점, 주기 운동, 카오스)를
3D 애니메이션과 함께 z(t) 곡선을 시각화하는 모듈
"""

import matplotlib
matplotlib.use('TkAgg')

# 수치 계산 및 시각화 라이브러리
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 로렌츠 방정식 및 수치 적분 함수 불러오기
from simulation.solver import integrate, lorenz_rhs

# 4가지 대표적인 Lorenz 시스템 상태의 파라미터 설정
settings = [
    {"sigma": 10, "rho": 0.5, "beta": 8/3, "label": "Converge to Origin"},     # 원점으로 수렴
    {"sigma": 10, "rho": 10, "beta": 8/3, "label": "Stable Fixed Point"},      # 정적인 평형점
    {"sigma": 10, "rho": 24, "beta": 8/3, "label": "Periodic Orbit"},          # 주기적 운동
    {"sigma": 10, "rho": 28, "beta": 8/3, "label": "Chaotic Butterfly"},       # 나비 모양의 카오스
]

# 공통 초기 조건과 시간 설정
y0 = np.array([1.0, 1.0, 1.0])  # x, y, z의 초기값
t0, t1, dt = 0.0, 50.0, 0.01    # 시뮬레이션 시간 구간과 시간 간격

# 결과 저장용 리스트
trajectories = []        # 전체 3차원 궤적
z_trajectories = []      # z(t) 만 따로 저장
ts = None                # 시간 배열

# 설정된 각 조건에 대해 시뮬레이션 수행
for s in settings:
    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y, sigma=s["sigma"], rho=s["rho"], beta=s["beta"]),
        y0=y0,
        t0=t0, t1=t1, dt=dt
    )
    trajectories.append(ys)
    z_trajectories.append(ys[:, 2])  # z 값만 추출하여 저장

# 전체 애니메이션 Figure 구성
fig = plt.figure(figsize=(16, 10))

# 3D 플롯 4개 생성 (2행 4열 중 상단에 배치)
axs_3d = [fig.add_subplot(2, 4, i + 1, projection='3d') for i in range(4)]

# z(t) 플롯은 하단 전체를 차지
ax_z = fig.add_subplot(2, 1, 2)

# 색상 지정
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# 각 플롯에 사용할 라인 및 포인트 객체 저장용 리스트
lines, points, z_lines = [], [], []

# 각 동역학 상태에 대해 3D 플롯 초기화
for i, ax in enumerate(axs_3d):
    s = settings[i]
    ax.set_title(s['label'])  # 플롯 제목: 동역학 상태
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    (line,) = ax.plot([], [], [], lw=1, color=colors[i])  # 궤적 선
    (point,) = ax.plot([], [], [], 'o', color=colors[i], markersize=3)  # 현재 위치 점
    lines.append(line)
    points.append(point)
    (zline,) = ax_z.plot([], [], label=s['label'], color=colors[i])  # z(t) 곡선
    z_lines.append(zline)

# z(t) 플롯 설정
ax_z.set_xlim(0, t1)
ax_z.set_ylim(0, 50)
ax_z.set_xlabel('Time (s)')
ax_z.set_ylabel('z(t)')
ax_z.set_title('z(t) Dynamics for Different Regimes')  # z값의 시간에 따른 변화 비교
ax_z.legend()

# 애니메이션 프레임 업데이트 함수 정의
def update(frame):
    for i in range(4):  # 4가지 경우 모두 반복
        data = trajectories[i][:frame]  # 현재까지의 시뮬레이션 데이터
        if data.shape[0] == 0:
            continue
        # 3D 궤적 업데이트
        lines[i].set_data(data[:, 0], data[:, 1])
        lines[i].set_3d_properties(data[:, 2])
        # 현재 위치 점 업데이트
        points[i].set_data([data[-1, 0]], [data[-1, 1]])
        points[i].set_3d_properties([data[-1, 2]])
        # z(t) 곡선 업데이트
        z_lines[i].set_data(ts[:frame], z_trajectories[i][:frame])
    return lines + points + z_lines

# 애니메이션 객체 생성
ani = FuncAnimation(
    fig, update,
    frames=range(0, len(ts), 2),  # 전체 프레임 수 (2개 간격으로 줄여서 속도 향상)
    interval=10,                  # 프레임 간 간격 (ms)
    blit=False
)

# 레이아웃 정리 및 애니메이션 시작
plt.tight_layout()
plt.show()