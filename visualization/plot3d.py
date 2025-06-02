"""
visualization/plot3d.py

Lorenz 어트랙터 3D 궤적을 시각화하는 모듈
"""
import numpy as np
import matplotlib
# IDE 호환 백엔드 설정
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.mplot3d import Axes3D  # 3D 그래프 지원

# 한글 폰트 설정 (전역 적용)
rcParams['font.family'] = 'AppleGothic'
rcParams['axes.unicode_minus'] = False


def plot_3d_trajectory(
    ys: np.ndarray,
    ts: np.ndarray = None,
    title: str = 'Lorenz 어트랙터 3D 플롯',
    colormap: str = 'viridis',
    figsize: tuple = (10, 7),
    marker_size: float = 0.5,
    alpha: float = 0.8,
    elev: float = 30,
    azim: float = 45,
    save_path: str = None,
    show: bool = True
) -> None:
    """
    주어진 Lorenz 궤적을 3D scatter로 시각화합니다.

    Args:
        ys: shape (N, 3) 배열, 시뮬레이션 결과 (x, y, z)
        ts: shape (N,) 배열, 시간축 (색상 매핑용, 없으면 인덱스 사용)
        title: 플롯 제목
        colormap: matplotlib 컬러맵 이름
        figsize: Figure 크기 (width, height)
        marker_size: scatter 마커 크기
        alpha: 마커 투명도
        elev: elevation 각도
        azim: azimuth 각도
        save_path: 파일 경로 지정 시 결과 이미지 저장
        show: True일 때 plt.show() 호출
    """
    if ts is None:
        ts = np.linspace(0, 1, ys.shape[0])
    else:
        # normalize time to [0,1]
        ts = (ts - ts.min()) / (ts.max() - ts.min())

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        ys[:, 0], ys[:, 1], ys[:, 2],
        c=ts, cmap=colormap,
        s=marker_size, alpha=alpha
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, label='Normalized Time')

    ax.set_xlabel('x (속도)')
    ax.set_ylabel('y (온도 분포)')
    ax.set_zlabel('z (대류 강도)')
    ax.set_title(title)

    ax.grid(True)
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    from simulation.solver import integrate, lorenz_rhs
    # 기본 파라미터
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    y0 = np.array([1.0, 1.0, 1.0])
    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta),
        y0=y0, t0=0.0, t1=50.0, dt=0.01
    )
    plot_3d_trajectory(
        ys=ys,
        ts=ts,
        title=f'Lorenz 어트랙터 (σ={sigma}, ρ={rho}, β={beta:.2f})',
        colormap='plasma',
        save_path=None,
        show=True
    )
