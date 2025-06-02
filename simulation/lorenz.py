"""
simulation/lorenz.py

Runge-Kutta solver를 사용한 Lorenz 어트랙터 시뮬레이션 실행 스크립트
"""
import matplotlib
# TkAgg 백엔드로 설정 (PyCharm 호환)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from simulation.solver import integrate, lorenz_rhs

# 한글 폰트 설정 (환경에 맞게 활성화)
mpl.rc('font', family='AppleGothic')   # macOS
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def main():
    """
    Lorenz 어트랙터 시뮬레이션을 수행하고 3D 그래프 출력
    """
    # 초기 조건 설정 (x0, y0, z0)
    y0 = np.array([1.0, 1.0, 1.0], dtype=float)
    # 시뮬레이션 시간 및 스텝 설정
    t0, t1, dt = 0.0, 50.0, 0.01

    # 통합 수행
    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y, sigma=10.0, rho=28.0, beta=8.0/3.0),
        y0=y0, t0=t0, t1=t1, dt=dt
    )

    # 3D 플롯
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ys[:,0], ys[:,1], ys[:,2], lw=0.5)

    # 레이블, 제목 설정
    ax.set_xlabel('x (속도)')
    ax.set_ylabel('y (온도 분포)')
    ax.set_zlabel('z (대류 강도)')
    ax.set_title('Lorenz 어트랙터 시뮬레이션')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
