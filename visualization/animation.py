"""
visualization/animation.py

Lorenz 어트랙터 3D 궤적 애니메이션 모듈
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # 3D 그래프 지원
from simulation.solver import integrate, lorenz_rhs

# 한글 폰트 설정 (환경에 맞게 변경하세요)
mpl.rc('font', family='AppleGothic')   # macOS

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False


def animate_lorenz(
    y0: np.ndarray,
    t0: float,
    t1: float,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    frames: int = None,
    interval: int = 30,
    save_path: str = None
) -> FuncAnimation:
    """
    Lorenz 어트랙터 시뮬레이션 후 3D 애니메이션 생성.

    Args:
        y0: 초기 상태 벡터 [x0, y0, z0]
        t0: 시작 시간
        t1: 종료 시간
        dt: 시간 간격
        sigma, rho, beta: Lorenz 시스템 파라미터
        frames: 프레임 수 (None이면 전체)
        interval: 프레임 간 인터벌(ms)
        save_path: 저장 경로 (.mp4)

    Returns:
        FuncAnimation 객체
    """
    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y, sigma, rho, beta),
        y0=y0, t0=t0, t1=t1, dt=dt
    )
    N = ys.shape[0]
    idxs = np.arange(N) if frames is None or frames > N else np.linspace(0, N-1, frames, dtype=int)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], lw=1)
    point, = ax.plot([], [], [], 'o', color='red', markersize=4)

    ax.set_xlim(ys[:,0].min(), ys[:,0].max())
    ax.set_ylim(ys[:,1].min(), ys[:,1].max())
    ax.set_zlim(ys[:,2].min(), ys[:,2].max())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz 어트랙터 애니메이션')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def update(i):
        idx = idxs[i]
        xs, ys_data, zs = ys[:idx+1,0], ys[:idx+1,1], ys[:idx+1,2]
        line.set_data(xs, ys_data)
        line.set_3d_properties(zs)
        point.set_data(xs[-1:], ys_data[-1:])
        point.set_3d_properties(zs[-1:])
        return line, point

    anim = FuncAnimation(fig, update, init_func=init, frames=len(idxs), interval=interval, blit=True)

    if save_path:
        writer = FFMpegWriter(fps=1000/interval)
        anim.save(save_path, writer=writer, dpi=200)
        plt.close(fig)
    else:
        plt.show()
    return anim


if __name__ == '__main__':
    y0 = np.array([1.0, 1.0, 1.0])
    animate_lorenz(
        y0, t0=0.0, t1=50.0, dt=0.01,
        frames=500, interval=20, save_path=None
    )