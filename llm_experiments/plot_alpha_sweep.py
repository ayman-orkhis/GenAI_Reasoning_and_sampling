#!/usr/bin/env python3
"""
Script pour tracer la Figure 6 style papier - Phi-3.5-mini-instruct
Usage: uv run python plot_figure6_style.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from grader_utils.math_grader import grade_answer


def eval_csv(filepath):
    """Évalue un fichier CSV et retourne l'accuracy MCMC."""
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath)
    total = len(df)
    mcmc_correct = 0
    
    for i in range(total):
        try:
            if grade_answer(str(df["mcmc_answer"][i]), str(df["correct_answer"][i])):
                mcmc_correct += 1
        except:
            pass
    
    return mcmc_correct / total * 100 if total > 0 else 0


# Configuration : vos fichiers (mcmc_steps=5)
configs = {
    1.0: "results/phi/phi_math_base_power_samp_results_5_1.0_0_0.csv",
    2.0: "results/phi/phi_math_base_power_samp_results_5_0.5_0_0.csv",
    4.0: "results/phi/phi_math_base_power_samp_results_5_0.25_0_0.csv",
    10.0: "results/phi/phi_math_base_power_samp_results_5_0.1_0_0.csv",
}

# Collecter les résultats
results = {}
print("📊 Évaluation des résultats...")

for alpha, filepath in configs.items():
    if os.path.exists(filepath):
        acc = eval_csv(filepath)
        results[alpha] = acc
        print(f"  α={alpha}: {acc:.1f}%")
    else:
        print(f"  α={alpha}: fichier non trouvé")

# ============================================================
# TRACER LE GRAPHIQUE STYLE FIGURE 6
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Données
alphas = [1.0, 2.0, 4.0, 10.0]
alpha_labels = ['α = 1.0', 'α = 2.0', 'α = 4.0', 'α = 10.0']

# Vos résultats
your_results = [results.get(a, 0) for a in alphas]

# Couleurs style papier (dégradé de bleu)
colors = ['#b3d4fc', '#6baed6', '#2171b5', '#08306b']

# Position des barres
x = np.arange(len(alphas))
width = 0.6

# Créer les barres
bars = ax.bar(x, your_results, width, color=colors, edgecolor='black', linewidth=0.8)

# Ajouter les valeurs au-dessus des barres
for bar, val in zip(bars, your_results):
    height = bar.get_height()
    ax.annotate(f'{val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

# Configuration des axes
ax.set_ylabel('MATH500 Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(alpha_labels, fontsize=11)
ax.set_ylim(0, 80)
ax.set_yticks(range(0, 81, 10))

# Grille horizontale légère
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

# Titre
ax.set_title('Phi-3.5-mini-instruct', fontsize=14, fontweight='bold', pad=10)

# Retirer les bordures
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Sauvegarder
os.makedirs("Results", exist_ok=True)
plt.savefig('Results/phi_figure6_style.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('Results/phi_figure6_style.pdf', dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n📊 Figure sauvegardée: Results/phi_figure6_style.png")

# Afficher le résumé comparatif
print("\n" + "="*45)
print("📋 COMPARAISON AVEC LE PAPIER")
print("="*45)
paper_ref = {1.0: 38.4, 2.0: 48.2, 4.0: 50.8, 10.0: 48.2}
print(f"{'α':<10} {'Vos résultats':<15} {'Papier':<10}")
print("-"*35)
for alpha in alphas:
    yours = results.get(alpha, 0)
    paper = paper_ref.get(alpha, 0)
    diff = yours - paper
    sign = "+" if diff >= 0 else ""
    print(f"α={alpha:<7} {yours:<15.1f} {paper:<10.1f} ({sign}{diff:.1f})")

plt.show()