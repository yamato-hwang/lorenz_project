"""
visualization/compare_initial_conditions.py

초기 조건이 서로 다른 Lorenz 시스템 세 가지를
동시에 3D 애니메이션으로 시각화하고,
세 궤적 사이의 거리 변화를 로그스케일로 함께 플로팅하는 시각화 모듈
"""

import matplotlib
matplotlib.use('TkAgg')

# 라이브러리
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Lorenz 시스템 정의 및 적분기 임포트
from simulation.solver import integrate, lorenz_rhs

# ✅ 서로 다른 세 가지 초기 조건
initial_conditions = [
    (0.00001, 0.00001, 0.00001),  # 아주 작은 값 → 원점 근처
    (1.0, 1.0, 1.0),              # 중간 값
    (5.0, 5.0, 5.0)               # 더 큰 값
]

# ✅ 공통 Lorenz 파라미터
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# ✅ 시간 범위 및 간격 설정
t0, t1, dt = 0.0, 50.0, 0.01

# ✅ 시뮬레이션 수행 결과 저장
trajectories = []
ts = None

for y0 in initial_conditions:
    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta),
        y0=np.array(y0),
        t0=t0, t1=t1, dt=dt
    )
    trajectories.append(ys)

# ✅ 두 궤적 간 유클리드 거리 계산
d12 = np.linalg.norm(trajectories[0] - trajectories[1], axis=1)
d13 = np.linalg.norm(trajectories[0] - trajectories[2], axis=1)
d23 = np.linalg.norm(trajectories[1] - trajectories[2], axis=1)

# ✅ 그림과 서브플롯 생성
fig = plt.figure(figsize=(15, 10))
axs = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(3)]  # 상단 3개의 3D 그래프
ax_dist = fig.add_subplot(2, 1, 2)  # 하단 거리 변화 그래프
colors = ['tab:blue', 'tab:orange', 'tab:green']

# ✅ 각 궤적에 대한 선(line)과 현재 점(point) 준비
lines = []
points = []
for i in range(3):
    ax = axs[i]
    ax.set_title(f"Initial: x0={initial_conditions[i][0]}")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    (line,) = ax.plot([], [], [], lw=0.8, color=colors[i])
    (point,) = ax.plot([], [], [], 'o', color=colors[i], markersize=3)
    lines.append(line)
    points.append(point)

# ✅ 거리 변화 그래프 설정 (로그스케일)
ax_dist.set_title("Distance between trajectories")
ax_dist.set_xlabel("Time (s)")
ax_dist.set_ylabel("Distance")
ax_dist.set_yscale('log')           # 로그 스케일로 변화 감지 용이
ax_dist.set_xlim(0, t1)
ax_dist.set_ylim(1e-8, 100)
line_d12, = ax_dist.plot([], [], label='0 vs 1')
line_d13, = ax_dist.plot([], [], label='0 vs 2')
line_d23, = ax_dist.plot([], [], label='1 vs 2')
ax_dist.legend()

# ✅ 프레임마다 업데이트 함수
def update(frame):
    for i in range(3):
        data = trajectories[i][:frame]
        if data.shape[0] == 0:
            continue
        lines[i].set_data(data[:, 0], data[:, 1])
        lines[i].set_3d_properties(data[:, 2])
        points[i].set_data([data[-1, 0]], [data[-1, 1]])
        points[i].set_3d_properties([data[-1, 2]])

    # 거리 변화 동시 업데이트
    line_d12.set_data(ts[:frame], d12[:frame])
    line_d13.set_data(ts[:frame], d13[:frame])
    line_d23.set_data(ts[:frame], d23[:frame])
    return lines + points + [line_d12, line_d13, line_d23]

# ✅ 애니메이션 생성
ani = FuncAnimation(
    fig, update,
    frames=range(0, len(ts), 2),  # 프레임 간격 2배로 → 속도 향상
    interval=10,                 # 프레임당 시간(ms)
    blit=False
)

plt.tight_layout()
plt.show()