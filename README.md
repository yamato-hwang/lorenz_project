# 🌪 Lorenz Attractor Simulation

This project simulates the **Lorenz attractor**, a canonical model of deterministic chaos. It supports numerical integration, quantitative chaos analysis, sensitivity testing, dynamic visualization, and GUI interaction.

## 🚀 Features

### ✅ Numerical Simulation & Modeling
- `simulation/solver.py`: RK4-based ODE integrator
- `simulation/lorenz.py`: Simulates Lorenz trajectories and static 3D visualization

### ✅ Chaos Quantification
- `simulation/lyapunov.py`: Estimates the **maximum Lyapunov exponent** using the Wolf method
- `simulation/fractal_dimension.py`: Computes **fractal dimension** via box-counting

### ✅ Sensitivity & Parameter Sweep
- `analysis/sensitivity.py`: Analyzes divergence from small initial perturbations
- `analysis/parameter_sweep.py`: Sweeps over parameters (σ, ρ, initial states) to quantify chaos levels

### ✅ Visualization & GUI
- `visualization/plot3d.py`: Time-colored 3D trajectories
- `visualization/animation.py`: Animated Lorenz attractor (GIF/MP4)
- `visualization/plot_stats.py`: Statistical plots (heatmaps, boxplots)
- `visualization/interactive_gui.py`: Interactive GUI using Streamlit

---

## 💻 Usage Examples

```bash
# (1) Run a basic simulation
python main.py simulate \
  --t0 0 --t1 50 --dt 0.01 \
  --x0 1 --y0 1 --z0 1 \
  --sigma 10 --rho 28 --beta 2.6667 \
  --save lorenz_plot.png

# (2) Generate animation
python main.py animate \
  --frames 500 --interval 20 \
  --save lorenz_anim.mp4

# (3) Estimate Lyapunov exponent
python main.py lyapunov \
  --t 100 --dt 0.01 --d0 1e-8 \
  --interval 0.1 --transient 5

# (4) Estimate fractal dimension
python main.py fractal \
  --scales 12 --save fractal_result.txt

# (5) Sensitivity analysis
python main.py sensitivity \
  --delta 1e-6 --save sensitivity.png

# (6) Launch GUI
streamlit run visualization/interactive_gui.py
# or
python main.py gui


⚙️ Customization
Change parameters (σ, ρ, β, T₁, dt) via CLI or GUI

Modify initial conditions in parameter_sweep.py

Adjust visualization styles in plot3d.py, animation.py

📖 Physical Interpretation
Lyapunov Exponent increases with ρ → higher unpredictability

Fractal Dimension ≈ 2.06 → trajectory lies on a complex surface

Butterfly Effect → Tiny initial differences → huge long-term divergence

🧑‍💻 Author
JeongMin Hwang (2025)
Based on the original Lorenz system (1963):
