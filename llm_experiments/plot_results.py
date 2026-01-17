# plot_figure6.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import sys
sys.path.append('.')

from grader_utils.math_grader import grade_answer

def safe_grade(ans, correct):
    """Grade une réponse de manière sécurisée"""
    try:
        return int(grade_answer(str(ans), str(correct)))
    except:
        return 0

def evaluate_csv(filepath):
    """Évalue un fichier CSV et retourne l'accuracy MCMC"""
    df = pd.read_csv(filepath)
    total = len(df)
    
    # Trouver les bonnes colonnes
    correct_col = None
    for col in ['correct_answer', 'answer']:
        if col in df.columns:
            correct_col = col
            break
    
    if correct_col is None:
        print(f"  Warning: No correct answer column in {filepath}")
        return None
    
    # Calculer les accuracies
    results = {}
    
    # MCMC accuracy (power sampling)
    if 'mcmc_answer' in df.columns:
        mcmc_correct = sum(safe_grade(df['mcmc_answer'][i], df[correct_col][i]) for i in range(total))
        results['mcmc'] = mcmc_correct / total * 100
    
    # Standard (base) accuracy
    if 'std_answer' in df.columns:
        std_correct = sum(safe_grade(df['std_answer'][i], df[correct_col][i]) for i in range(total))
        results['std'] = std_correct / total * 100
    
    # Naive temperature accuracy
    if 'naive_answer' in df.columns:
        naive_correct = sum(safe_grade(df['naive_answer'][i], df[correct_col][i]) for i in range(total))
        results['naive'] = naive_correct / total * 100
    
    return results

def load_alpha_results(base_path="Results/figure6_alpha"):
    """Charge les résultats pour différentes valeurs de α"""
    results = {}
    
    model_names = {
        "qwen_math": "Qwen2.5-Math-7B",
        "qwen": "Qwen2.5-7B",
        "phi": "Phi-3.5-mini-instruct"
    }
    
    for model_key, model_name in model_names.items():
        results[model_name] = {}
        
        for alpha in [1.0, 2.0, 4.0, 10.0]:
            folder = Path(base_path) / f"{model_key}_alpha_{alpha}"
            csv_files = list(folder.glob("*.csv")) if folder.exists() else []
            
            if csv_files:
                eval_result = evaluate_csv(str(csv_files[0]))
                if eval_result and 'mcmc' in eval_result:
                    results[model_name][alpha] = eval_result['mcmc']
                    print(f"  {model_name} α={alpha}: {eval_result['mcmc']:.1f}%")
    
    return results

def load_mcmc_results(base_path="Results/figure6_mcmc"):
    """Charge les résultats pour différents nombres de MCMC steps"""
    results = {}
    
    model_names = {
        "qwen_math": "Qwen2.5-Math-7B",
        "qwen": "Qwen2.5-7B"
    }
    
    for model_key, model_name in model_names.items():
        results[model_name] = {}
        
        for steps in [0, 2, 4, 6, 8, 10]:
            folder = Path(base_path) / f"{model_key}_mcmc_{steps}"
            csv_files = list(folder.glob("*.csv")) if folder.exists() else []
            
            if csv_files:
                eval_result = evaluate_csv(str(csv_files[0]))
                if eval_result and 'mcmc' in eval_result:
                    results[model_name][steps] = eval_result['mcmc']
                    print(f"  {model_name} mcmc={steps}: {eval_result['mcmc']:.1f}%")
    
    return results

def plot_figure6(alpha_results, mcmc_results, output_file="figure6_reproduction.png"):
    """Trace la Figure 6"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== LEFT: Effect of α ==========
    ax1 = axes[0]
    
    models = ['Qwen2.5-Math-7B', 'Qwen2.5-7B', 'Phi-3.5-mini-instruct']
    alphas = [1.0, 2.0, 4.0, 10.0]
    x = np.arange(len(models))
    width = 0.18
    
    colors = ['#b3d9ff', '#66b3ff', '#3399ff', '#0066cc']
    
    for i, alpha in enumerate(alphas):
        values = []
        for model in models:
            if model in alpha_results and alpha in alpha_results[model]:
                values.append(alpha_results[model][alpha])
            else:
                values.append(0)
        
        bars = ax1.bar(x + i*width, values, width, 
                      label=f'α = {alpha}', color=colors[i], 
                      edgecolor='black', linewidth=0.5)
        
        # Ajouter les valeurs
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('MATH500 Accuracy (%)', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 85)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_title('Effect of α', fontsize=13, fontweight='bold')
    
    # ========== RIGHT: Effect of MCMC Steps ==========
    ax2 = axes[1]
    
    colors_line = ['#1f78b4', '#ff7f0e']
    markers = ['o', 's']
    
    for (model, data), color, marker in zip(mcmc_results.items(), colors_line, markers):
        if data:
            steps = sorted(data.keys())
            accuracies = [data[s] for s in steps]
            ax2.plot(steps, accuracies, marker=marker, color=color, 
                    label=model, linewidth=2, markersize=8)
    
    ax2.set_xlabel('MCMC Steps', fontsize=12)
    ax2.set_ylabel('MATH500 Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(60, 80)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks([0, 2, 4, 6, 8, 10])
    ax2.set_title('Effect of MCMC Steps', fontsize=13, fontweight='bold')
    
    plt.suptitle('Figure 6: Effect of hyperparameters on power sampling', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Figure saved to {output_file}")


if __name__ == "__main__":
    print("="*60)
    print("Loading Alpha Results...")
    print("="*60)
    alpha_results = load_alpha_results()
    
    print("\n" + "="*60)
    print("Loading MCMC Results...")
    print("="*60)
    mcmc_results = load_mcmc_results()
    
    print("\n" + "="*60)
    print("Plotting Figure 6...")
    print("="*60)
    plot_figure6(alpha_results, mcmc_results)