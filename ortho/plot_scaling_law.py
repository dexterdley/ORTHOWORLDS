import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

# Data: Number of game environments vs. Model performance metrics
num_games = np.array([1, 5, 10, 25, 50, 75, 100])

# Simulated scaling law performance (PSNR in dB & Zero-Shot Transfer %)
psnr_scores = np.array([18.2, 22.5, 25.8, 29.4, 32.1, 33.8, 35.2])
zero_shot_acc = np.array([12.5, 34.0, 52.8, 71.5, 83.2, 89.0, 92.5])

# Plot PSNR Curve (Primary Y-axis)
color1 = '#38bdf8' # Bright cyan
line1 = ax.plot(num_games, psnr_scores, 'o-', color=color1, linewidth=2.5, markersize=8, label='Multi-View Reconstruction (PSNR dB)')
ax.set_xlabel('Number of Training Game Environments', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Visual Fidelity (PSNR in dB)', fontsize=12, fontweight='bold', labelpad=10)
ax.tick_params(axis='y')

# Plot Zero-Shot Generalization (Secondary Y-axis)
ax2 = ax.twinx()
color2 = '#c084fc' # Bright purple
line2 = ax2.plot(num_games, zero_shot_acc, 's--', color=color2, linewidth=2.5, markersize=8, label='Zero-Shot Generalization Score (%)')
ax2.set_ylabel('Zero-Shot Physics Generalization (%)', fontsize=12, fontweight='bold', labelpad=10)
ax2.tick_params(axis='y')

# Grid and styling
ax.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')
ax.set_title('Scaling Laws\nPerformance vs. Number of Training Environments', 
             fontsize=14, fontweight='bold', pad=15)

# Highlight saturation threshold
ax.axvline(x=50, linestyle=':', alpha=0.6)
ax.annotate('Diminishing Return Frontier (~50 Games)', xy=(50, 32.1), xytext=(55, 26.0),
            arrowprops=dict(facecolor='#818cf8', shrink=0.08, width=1.5, headwidth=8),
            fontsize=10, fontweight='bold')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='lower right', frameon=True)

plt.tight_layout()

# Save plot
output_path = "assets/scaling_law_envs.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Plot saved successfully to {output_path}")

# plt.show()

