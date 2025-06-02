import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_heatmap(csv_path, value_col, x, y, save_path=None):
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index=y, columns=x, values=value_col)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
    plt.title(f"Heatmap of {value_col}")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_boxplot(csv_path, value_col, group_col, save_path=None):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=group_col, y=value_col, data=df)
    plt.title(f"{value_col} Distribution by {group_col}")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
