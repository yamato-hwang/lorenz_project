"""
visualization/plot3d.py

Lorenz 어트랙터 3D 궤적을 시각화하는 모듈
"""

import matplotlib
# IDE 호환을 위해 TkAgg 백엔드 사용
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 한글 폰트 및 마이너스 기호 설정
mpl.rc('font', family='AppleGothic')   # macOS
mpl.rcParams['axes.unicode_minus'] = False


def plot_3d_trajectory(
    ys: np.ndarray,
    title: str = 'Lorenz 어트랙터 3D 플롯',
    colormap: str = 'viridis',
    show: bool = True,
    save_path: str = None
) -> None:
    """
    주어진 Lorenz 궤적 데이터를 3차원 그래프로 표시.

    Args:
        ys: (N,3) 배열, 시뮬레이션 결과 [x, y, z]
        title: 플롯 제목
        colormap: matplotlib 컬러맵 이름
        show: True면 plt.show() 호출
        save_path: 파일 경로 지정 시 이미지 저장
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    t = np.linspace(0, 1, ys.shape[0])
    ax.scatter(
        ys[:, 0], ys[:, 1], ys[:, 2],
        c=t, cmap=colormap, s=0.5, alpha=0.8
    )

    ax.set_xlabel('x (속도)')
    ax.set_ylabel('y (온도 분포)')
    ax.set_zlabel('z (대류 강도)')
    ax.set_title(title)
    ax.grid(True)
    ax.view_init(elev=30, azim=45)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    from simulation.solver import integrate, lorenz_rhs

    # 초기 조건 및 통합
    y0 = np.array([1.0, 1.0, 1.0])
    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y),
        y0=y0, t0=0.0, t1=50.0, dt=0.01
    )

    plot_3d_trajectory(
        ys,
        title='Lorenz 어트랙터 궤적',
        colormap='plasma',
        show=True,
        save_path=None
    )
