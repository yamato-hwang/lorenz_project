def max_lyapunov_exponent():
    """
    simulation/lyapunov.py

    Lorenz 시스템의 최대 Lyapunov 지수를 Wolf 방법으로 추정하는 모듈
    """

    import numpy as np
    from typing import Callable
    from simulation.solver import rk4_step

    def max_lyapunov_exponent(
            f: Callable[[float, np.ndarray], np.ndarray],
            y0: np.ndarray,
            dt: float,
            T: float,
            d0: float = 1e-8,
            renorm_interval: float = 0.1,
            transient: float = 5.0
    ) -> float:
        """
        Wolf 방법을 사용하여 최대 Lyapunov 지수를 계산합니다.

        Parameters:
        - f: 함수, t와 상태 y를 입력받아 dy/dt를 반환
        - y0: 초기 상태 벡터
        - dt: 시뮬레이션 시간 간격
        - T: 전체 시뮬레이션 시간
        - d0: 초기 교란 크기 (기본값 1e-8)
        - renorm_interval: 재정규화 간격 (기본값 0.1)
        - transient: 과도 상태 무시 시간 (기본값 5.0)

        Returns:
        - 최대 Lyapunov 지수
        """
        total_steps = int(np.ceil(T / dt))
        renorm_steps = max(int(np.round(renorm_interval / dt)), 1)

        y = y0.astype(float).copy()
        perturb = np.random.randn(*y0.shape)
        perturb /= np.linalg.norm(perturb)
        y_pert = y + d0 * perturb

        sum_log = 0.0
        count = 0

        for step in range(1, total_steps + 1):
            t = step * dt
            y = rk4_step(f, (step - 1) * dt, y, dt)
            y_pert = rk4_step(f, (step - 1) * dt, y_pert, dt)

            if t > transient and step % renorm_steps == 0:
                diff = y_pert - y
                dist = np.linalg.norm(diff)
                sum_log += np.log(dist / d0)
                count += 1
                y_pert = y + (d0 * diff / dist)

        if count == 0:
            raise RuntimeError(
                "재정규화 단계가 실행되지 않았습니다. transient 또는 renorm_interval을 조정하세요."
            )

        total_time = count * renorm_interval
        return sum_log / total_time

    if __name__ == "__main__":
        from simulation.solver import lorenz_rhs

        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        y0 = np.array([1.0, 1.0, 1.0])

        def f(t, y):
            return lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta)

        lmax = max_lyapunov_exponent(
            f=f,
            y0=y0,
            dt=0.01,
            T=100.0,
            d0=1e-8,
            renorm_interval=0.1,
            transient=5.0
        )
        print(f"Estimated max Lyapunov exponent: {lmax:.6f}")
