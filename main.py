#!/usr/bin/env python3
# main.py

"""
CLI를 통해 다양한 Lorenz 어트랙터 분석 및 시각화 기능을 제공합니다.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# 프로젝트 루트 기준 결과 저장 디렉터리 설정
OUTPUT_DIR = Path('results/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from simulation.solver import integrate, lorenz_rhs
from visualization.plot3d import plot_3d_trajectory
from visualization.animation import animate_lorenz
from simulation.lyapunov import max_lyapunov_exponent
from simulation.fractal_dimension import box_counting_dimension
from analysis.sensitivity import sensitivity_analysis, plot_sensitivity


def main():
    parser = argparse.ArgumentParser(description='Lorenz 어트랙터 분석 도구')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # simulate
    sim = subparsers.add_parser('simulate', help='3D 궤적 시뮬레이션 및 정적 플롯')
    sim.add_argument('--t0', type=float, default=0.0)
    sim.add_argument('--t1', type=float, default=50.0)
    sim.add_argument('--dt', type=float, default=0.01)
    sim.add_argument('--x0', type=float, default=1.0)
    sim.add_argument('--y0', type=float, default=1.0)
    sim.add_argument('--z0', type=float, default=1.0)
    sim.add_argument('--sigma', type=float, default=10.0)
    sim.add_argument('--rho', type=float, default=28.0)
    sim.add_argument('--beta', type=float, default=8.0/3.0)
    sim.add_argument('--save', type=str, default=None, help='저장 파일 이름 (.png)')

    # animate
    anim = subparsers.add_parser('animate', help='3D 궤적 애니메이션')
    anim.add_argument('--t0', type=float, default=0.0)
    anim.add_argument('--t1', type=float, default=50.0)
    anim.add_argument('--dt', type=float, default=0.01)
    anim.add_argument('--sigma', type=float, default=10.0)
    anim.add_argument('--rho', type=float, default=28.0)
    anim.add_argument('--beta', type=float, default=8.0/3.0)
    anim.add_argument('--frames', type=int, default=500)
    anim.add_argument('--interval', type=int, default=20)
    anim.add_argument('--save', type=str, default=None, help='저장 파일 이름 (.mp4)')

    # lyapunov
    lyap = subparsers.add_parser('lyapunov', help='최대 Lyapunov 지수 계산')
    lyap.add_argument('--dt', type=float, default=0.01)
    lyap.add_argument('--T', type=float, default=100.0)
    lyap.add_argument('--d0', type=float, default=1e-8)
    lyap.add_argument('--interval', type=float, default=0.1)
    lyap.add_argument('--transient', type=float, default=5.0)

    # fractal
    frac = subparsers.add_parser('fractal', help='프랙탈 차원 추정')
    frac.add_argument('--scales', type=int, default=12)
    frac.add_argument('--save', type=str, default=None, help='저장 파일 이름 (.txt)')

    # sensitivity
    sens = subparsers.add_parser('sensitivity', help='초기 조건 민감도 분석')
    sens.add_argument('--delta', type=float, default=1e-6)
    sens.add_argument('--t0', type=float, default=0.0)
    sens.add_argument('--t1', type=float, default=50.0)
    sens.add_argument('--dt', type=float, default=0.01)
    sens.add_argument('--sigma', type=float, default=10.0)
    sens.add_argument('--rho', type=float, default=28.0)
    sens.add_argument('--beta', type=float, default=8.0/3.0)
    sens.add_argument('--save', type=str, default=None, help='저장 파일 이름 (.png)')

    # gui
    subparsers.add_parser('gui', help='Streamlit GUI 실행 안내')

    args = parser.parse_args()

    # 공통 초기 조건
    y0 = np.array([getattr(args, 'x0', 1.0),
                   getattr(args, 'y0', 1.0),
                   getattr(args, 'z0', 1.0)])

    if args.command == 'simulate':
        ts, ys = integrate(
            f=lambda t,y: lorenz_rhs(t, y, args.sigma, args.rho, args.beta),
            y0=y0, t0=args.t0, t1=args.t1, dt=args.dt
        )
        save_path = OUTPUT_DIR/args.save if args.save else None
        plot_3d_trajectory(ys, save_path=str(save_path) if save_path else None)

    elif args.command == 'animate':
        save_path = OUTPUT_DIR/args.save if args.save else None
        animate_lorenz(
            y0=y0, t0=args.t0, t1=args.t1, dt=args.dt,
            sigma=args.sigma, rho=args.rho, beta=args.beta,
            frames=args.frames, interval=args.interval,
            save_path=str(OUTPUT_DIR/args.save) if args.save else None
        )

    elif args.command == 'lyapunov':
        f = lambda t,y: lorenz_rhs(t, y, args.sigma, args.rho, args.beta)
        lmax = max_lyapunov_exponent(
            f=f, y0=y0, dt=args.dt, T=args.T,
            d0=args.d0, renorm_interval=args.interval,
            transient=args.transient
        )
        print(f"Estimated max Lyapunov exponent: {lmax:.6f}")

    elif args.command == 'fractal':
        ts, ys = integrate(lambda t,y: lorenz_rhs(t,y,args.sigma,args.rho,args.beta),
                           y0=y0, t0=0.0, t1=50.0, dt=0.01)
        sample = ys[int(5.0/0.01)::10]
        eps, cnt, dim = box_counting_dimension(sample, n_scales=args.scales)
        output = f"Estimated fractal dimension: {dim:.4f}\n"
        print(output)
        if args.save:
            path = OUTPUT_DIR/args.save
            with open(path, 'w') as f:
                f.write('epsilons: ' + str(eps) + '\n')
                f.write('counts: ' + str(cnt) + '\n')
                f.write(output)

    elif args.command == 'sensitivity':
        ts, distances = sensitivity_analysis(
            y0=y0, delta=args.delta,
            t0=args.t0, t1=args.t1, dt=args.dt,
            sigma=args.sigma, rho=args.rho, beta=args.beta
        )
        save_path = OUTPUT_DIR/args.save if args.save else None
        plot_sensitivity(ts, distances, save_path=str(save_path) if save_path else None)

    elif args.command == 'gui':
        sys.exit("Streamlit 앱은 다음 명령어로 실행하세요: streamlit run visualization/interactive_gui.py")


if __name__ == '__main__':
    main()
