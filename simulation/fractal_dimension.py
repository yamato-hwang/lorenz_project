"""
simulation/fractal_dimension.py

박스 카운팅(Box-Counting) 기법을 이용해 프랙탈 차원을 추정하는 모듈
"""

import numpy as np
import matplotlib
# IDE 호환 백엔드 설정
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정
mpl.rc('font', family='AppleGothic')   # macOS
# mpl.rc('font', family='Malgun Gothic')  # Windows
# mpl.rc('font', family='NanumGothic')   # Linux
mpl.rcParams['axes.unicode_minus'] = False

from typing import Tuple, Optional


def box_counting_dimension(
    points: np.ndarray,
    n_scales: int = 12,
    scale_range: Optional[Tuple[int, int]] = None,
    lower_frac: float = 0.01,
    upper_frac: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    주어진 점 구름에 대해 박스 카운팅 방식으로 프랙탈 차원을 추정합니다.

    Parameters:
    - points: np.ndarray, shape (N, D) 점 데이터
    - n_scales: int, 사용할 스케일 개수
    - scale_range: Tuple(start, end), 수동 회귀 인덱스 범위
                   None이면 자동 탐색
    - lower_frac, upper_frac: 자동 탐색 시 count 비율 임계값

    Returns:
    - epsilons: np.ndarray, 각 스케일의 박스 크기
    - counts: np.ndarray, 각 스케일에서 사용된 박스 개수
    - dimension: float, 추정된 프랙탈 차원
    - used_range: Tuple(lo, hi), 회귀에 사용된 인덱스 범위
    """
    if points.size == 0:
        raise ValueError("점 구름 데이터가 비어 있습니다.")
    if n_scales < 2:
        raise ValueError("스케일 개수(n_scales)는 최소 2 이상이어야 합니다.")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    max_span = (maxs - mins).max()

    epsilons = max_span / (2 ** np.arange(1, n_scales + 1))
    counts = np.array([
        np.unique(np.floor((points - mins) / eps).astype(int), axis=0).shape[0]
        for eps in epsilons
    ])

    logs = np.log(counts)
    log_inv_eps = np.log(1.0 / epsilons)

    # 회귀 범위 결정 및 자동 탐색 (최대 R² 기준 슬라이딩 윈도우)
    if scale_range is None:
        best_r2 = -np.inf
        best_lo, best_hi = 1, n_scales
        min_win = max(3, n_scales // 4)
        y_all = logs
        x_all = log_inv_eps
        for lo_candidate in range(1, n_scales - min_win + 1):
            for hi_candidate in range(lo_candidate + min_win, n_scales + 1):
                x_win = x_all[lo_candidate:hi_candidate]
                y_win = y_all[lo_candidate:hi_candidate]
                coeffs_win, residuals, *_ = np.polyfit(x_win, y_win, 1, full=True)
                if residuals.size > 0:
                    SS_res = residuals[0]
                else:
                    y_pred = coeffs_win[0] * x_win + coeffs_win[1]
                    SS_res = np.sum((y_win - y_pred) ** 2)
                SS_tot = np.sum((y_win - y_win.mean()) ** 2)
                r2 = 1 - SS_res / SS_tot if SS_tot > 0 else -np.inf
                if r2 > best_r2:
                    best_r2 = r2
                    best_lo, best_hi = lo_candidate, hi_candidate
        lo, hi = best_lo, best_hi
    else:
        lo, hi = scale_range

    lo = max(lo, 1)
    hi = min(hi, n_scales)

    # 선택된 구간으로 최종 회귀 수행
    final_coeffs = np.polyfit(log_inv_eps[lo:hi], logs[lo:hi], 1)
    dimension = float(final_coeffs[0])

    return epsilons, counts, dimension, (lo, hi)


def plot_loglog(
    epsilons: np.ndarray,
    counts: np.ndarray,
    used_range: Tuple[int, int]
) -> None:
    """
    로그-로그 플롯 및 회귀 범위 표시
    """
    inv_eps = 1.0 / epsilons
    lo, hi = used_range
    plt.figure(figsize=(6, 5))
    plt.loglog(inv_eps, counts, 'o-', label='Data')
    plt.loglog(inv_eps[lo:hi], counts[lo:hi], 'ro-', label='Regression region')
    plt.xlabel('1/ε')
    plt.ylabel('N(ε)')
    plt.title('Box-Counting Log-Log Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()


def example():
    """
    Lorenz 궤적을 이용한 프랙탈 차원 추정 예제
    """
    from simulation.solver import integrate, lorenz_rhs

    ts, ys = integrate(
        f=lambda t, y: lorenz_rhs(t, y),
        y0=np.array([1.0, 1.0, 1.0]),
        t0=0.0, t1=50.0, dt=0.01
    )
    sample = ys[int(5.0 / 0.01):]

    eps, cnt, dim, used = box_counting_dimension(sample)
    lo, hi = used
    print(f"Selected scales indices for regression: {lo}~{hi-1}")
    print("epsilons:", eps)
    print("counts:", cnt)
    print(f"Estimated fractal dimension: {dim:.4f}\n")

    plot_loglog(eps, cnt, used)

if __name__ == '__main__':
    example()