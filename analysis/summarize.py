"""
summarize.py

parameter_sweep_results.csv를 불러와
sigma, rho, x0별로 Lyapunov 지수와 Fractal 차원의
평균, 표준편차, 95% 신뢰구간을 계산하여
summary_results.csv로 저장하는 스크립트
"""

import pandas as pd
import numpy as np
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def summarize(
    input_csv: str = 'results/parameter_sweep_results.csv',
    output_csv: str = 'results/summary_results.csv'
) -> None:
    if not os.path.exists(input_csv):
        logging.error(f"Input file '{input_csv}' not found.")
        return

    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} rows from {input_csv}")

    # 유효한 값만 필터링 (NaN 제거)
    df_clean = df.dropna(subset=['lyapunov_exponent', 'fractal_dimension'])
    if df_clean.empty:
        logging.warning("No valid data found after dropping NaNs.")
        return

    grouped = df_clean.groupby(['sigma', 'rho', 'x0'])
    stats = grouped.agg(
        lyap_mean=('lyapunov_exponent', 'mean'),
        lyap_std=('lyapunov_exponent', 'std'),
        fd_mean=('fractal_dimension', 'mean'),
        fd_std=('fractal_dimension', 'std'),
        count=('lyapunov_exponent', 'count')
    ).reset_index()

    z = 1.96  # 95% 신뢰수준
    with np.errstate(divide='ignore', invalid='ignore'):
        stats['lyap_ci_lower'] = stats['lyap_mean'] - z * stats['lyap_std'] / stats['count'].pow(0.5)
        stats['lyap_ci_upper'] = stats['lyap_mean'] + z * stats['lyap_std'] / stats['count'].pow(0.5)
        stats['fd_ci_lower']   = stats['fd_mean']  - z * stats['fd_std']  / stats['count'].pow(0.5)
        stats['fd_ci_upper']   = stats['fd_mean']  + z * stats['fd_std']  / stats['count'].pow(0.5)

    stats.to_csv(output_csv, index=False)
    logging.info(f"Saved summary statistics to {output_csv} ({len(stats)} rows)")

if __name__ == '__main__':
    summarize()
