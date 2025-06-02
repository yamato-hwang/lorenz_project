"""
analysis/sensitivity.py

Lorenz 시스템 초기 조건 민감도 분석 모듈
두 궤적을 약간 다른 초기 조건으로 시뮬레이션하여
시간에 따른 두 상태 간 거리(발산 속도)를 계산 및 시각화합니다.
"""
import numpy as np
import matplotlib
# Headless 환경에서도 그래프 저장 가능하도록 non-interactive 백엔드 사용
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional
from simulation.solver import integrate, lorenz_rhs

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def sensitivity_analysis(
    y0: np.ndarray,
    delta: float = 1e-6,
    t0: float = 0.0,
    t1: float = 50.0,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    초기 조건 민감도 분석 수행.

    Args:
        y0: 기준 초기 상태 벡터 [x0, y0, z0]
        delta: 초기 교란 크기
        t0: 시작 시간
        t1: 종료 시간
        dt: 시간 간격
        sigma, rho, beta: Lorenz 시스템 파라미터

    Returns:
        ts: 시간 배열
        distances: 각 시간에서 두 궤적 상태 간 거리 배열
    """
    logger.info(f"Starting sensitivity analysis: y0={y0}, delta={delta}")
    # 기준 및 교란 초기 조건
    y0_ref = y0.astype(float)
    y0_pert = y0_ref + np.array([delta, 0.0, 0.0])

    # 시뮬레이션 수행
    ts, ys_ref = integrate(
        lambda t, y: lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta),
        y0=y0_ref, t0=t0, t1=t1, dt=dt
    )
    _, ys_pert = integrate(
        lambda t, y: lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta),
        y0=y0_pert, t0=t0, t1=t1, dt=dt
    )

    # 거리 계산
    distances = np.linalg.norm(ys_pert - ys_ref, axis=1)
    logger.info("Sensitivity analysis completed.")
    return ts, distances


def plot_sensitivity(
    ts: np.ndarray,
    distances: np.ndarray,
    log_scale: bool = True,
    title: str = '초기 조건 민감도 분석',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> None:
    """
    민감도 분석 결과 거리 데이터를 플로팅 및 저장.

    Args:
        ts: 시간 배열
        distances: 거리 배열
        log_scale: True면 y축 로그 스케일 사용
        title: 플롯 제목
        figsize: 그림 크기
        save_path: 파일 경로 지정 시 이미지 저장 (None일 경우 'sensitivity.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    if log_scale:
        ax.set_yscale('log')
    ax.plot(ts, distances, lw=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance between trajectories')
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 이미지 저장
    if save_path is None:
        save_path = 'sensitivity.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sensitivity plot saved to {save_path}")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lorenz 시스템 초기 조건 민감도 분석')
    parser.add_argument('--x0', type=float, default=1.0)
    parser.add_argument('--y0', type=float, default=1.0)
    parser.add_argument('--z0', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-6)
    parser.add_argument('--t0', type=float, default=0.0)
    parser.add_argument('--t1', type=float, default=50.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--rho', type=float, default=28.0)
    parser.add_argument('--beta', type=float, default=8.0/3.0)
    parser.add_argument('--no-log', action='store_true', help='Disable log scale on y-axis')
    parser.add_argument('--save', type=str, help='Path to save plot image')
    args = parser.parse_args()

    y0 = np.array([args.x0, args.y0, args.z0])
    ts, distances = sensitivity_analysis(
        y0, delta=args.delta,
        t0=args.t0, t1=args.t1, dt=args.dt,
        sigma=args.sigma, rho=args.rho, beta=args.beta
    )
    plot_sensitivity(
        ts, distances,
        log_scale=not args.no_log,
        title=f'Sensitivity Analysis (Δ={args.delta})',
        save_path=args.save
    )

if __name__ == '__main__':
    main()
