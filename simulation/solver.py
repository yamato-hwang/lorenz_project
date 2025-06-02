'''
simulation/solver.py

Runge-Kutta 4차 방법 ODE 계산기 및 Lorenz 시스템 우변 구현
'''

from typing import Callable, Tuple
import numpy as np


def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    단일 Runge-Kutta 4차(RK4) 스텝을 수행하여 상태를 dt만큼 전진.

    Args:
        f: ODE 우변 함수 f(t, y) -> dy/dt (ndarray)
        t: 현재 시간
        y: 현재 상태 벡터
        dt: 시간 간격

    Returns:
        다음 상태 벡터 (ndarray)
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t0: float,
    t1: float,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    고정 스텝 RK4를 이용해 ODE y' = f(t, y)를 t0에서 t1까지 적분.

    Args:
        f: ODE 우변 함수
        y0: 시작 상태 벡터
        t0: 시작 시간
        t1: 종료 시간
        dt: 시간 간격

    Returns:
        ts: 시간 배열 (shape=(N_steps+1,))
        ys: 상태 배열 (shape=(N_steps+1, len(y0)))
    """
    if dt <= 0:
        raise ValueError("시간 간격 dt는 양수여야 합니다.")
    total_steps = int(np.ceil((t1 - t0) / dt))
    ts = np.linspace(t0, t0 + total_steps * dt, total_steps + 1)
    ys = np.zeros((total_steps + 1, y0.size))
    ys[0] = y0
    for i in range(total_steps):
        ys[i + 1] = rk4_step(f, ts[i], ys[i], dt)
    return ts, ys


def lorenz_rhs(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0
) -> np.ndarray:
    """
    Lorenz 시스템 우변을 계산.

    방정식:
        dx/dt = sigma * (y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z

    Args:
        t: 시간 (호환성 용도, 사용 안 함)
        state: [x, y, z] 상태 벡터
        sigma: 프란델 수
        rho: 레일리 수
        beta: 기하 계수

    Returns:
        [dx/dt, dy/dt, dz/dt] (ndarray)
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)
