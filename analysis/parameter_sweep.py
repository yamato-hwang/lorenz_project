"""
analysis/parameter_sweep.py

다양한 파라미터와 초기 조건에 따라 Lorenz 시스템을 시뮬레이션하고
Lyapunov 지수 및 프랙탈 차원을 계산하여 CSV로 저장하는 스크립트
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import traceback
import logging

from simulation.solver import integrate, lorenz_rhs
from simulation.lyapunov import max_lyapunov_exponent
from simulation.fractal_dimension import box_counting_dimension

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 파라미터 스윕 설정
SIGMAS = [5.0, 10.0, 20.0]       # σ (Prandtl 수)
RHOS   = [20.0, 28.0, 35.0]     # ρ (Rayleigh 수)
BETA   = 8.0 / 3.0              # β (Geometry factor)

# 초기 조건 스위프 (x0, y0, z0)
INITIAL_CONDITIONS = [
    (0.1, 0.1, 0.1),
    (1.0, 1.0, 1.0),
    (5.0, 5.0, 5.0)
]

# 반복 횟수 설정 (통계량 계산을 위해)
REPEATS = 5

# 시뮬레이션 타임 설정
T0 = 0.0
T1 = 100.0  # 더 긴 시뮬레이션으로 안정성 확보
DT = 0.01

# Lyapunov 계산 설정
LYAP_D0 = 1e-8
LYAP_RENORM = 0.1
LYAP_TRANSIENT = 10.0  # 더 긴 과도 상태 제거

# 결과 저장 리스트
records = []

# 전체 파라미터 조합에 대해 반복
for sigma in SIGMAS:
    for rho in RHOS:
        for x0, y0, z0 in INITIAL_CONDITIONS:
            for repeat in range(REPEATS):
                y0_vec = np.array([x0, y0, z0], dtype=float)
                logging.info(f"[R{repeat+1}] Simulating: sigma={sigma}, rho={rho}, x0={x0}")

                try:
                    ts, ys = integrate(
                        f=lambda t, y: lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=BETA),
                        y0=y0_vec,
                        t0=T0,
                        t1=T1,
                        dt=DT
                    )
                except Exception as e:
                    logging.error(f"Integration failed: {e}")
                    traceback.print_exc()
                    continue

                try:
                    lyap = max_lyapunov_exponent(
                        rhs=lambda t, y: lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=BETA),
                        y0=y0_vec,
                        dt=DT,
                        T=T1,
                        d0=LYAP_D0,
                        renorm_interval=LYAP_RENORM,
                        transient=LYAP_TRANSIENT
                    )
                except Exception as e:
                    logging.warning(f"Lyapunov failed (sigma={sigma}, rho={rho}, x0={x0}): {e}")
                    traceback.print_exc()
                    lyap = np.nan

                sample = ys[int(LYAP_TRANSIENT/DT)::5]  # 더 촘촘한 샘플링
                try:
                    eps, cnt, fd, _ = box_counting_dimension(sample)
                except Exception as e:
                    logging.warning(f"Fractal dimension failed (sigma={sigma}, rho={rho}, x0={x0}): {e}")
                    traceback.print_exc()
                    fd = np.nan

                records.append({
                    'sigma': sigma,
                    'rho': rho,
                    'beta': BETA,
                    'x0': x0,
                    'y0': y0,
                    'z0': z0,
                    'repeat': repeat + 1,
                    'lyapunov_exponent': lyap,
                    'fractal_dimension': fd
                })

# DataFrame 생성 및 저장
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(records)
df.to_csv('results/parameter_sweep_results.csv', index=False)
logging.info(f"Saved results to results/parameter_sweep_results.csv ({len(df)} rows)")
