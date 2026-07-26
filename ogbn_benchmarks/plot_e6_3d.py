"""E6 scalability 3D: x=peak memory, y=train time, z=Test F1. One point per method.
F1 from E1 @0.4 (main comparison); memory + time from E6 run."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# model: (peak GPU GB, train time s, test F1 @0.4)
data = {
    'NeighAggre': (0.48, 46,   0.325),
    'KNN':        (0.26, 49,   0.335),
    'GraphSAGE':  (0.14, 359,  0.372),
    'FP':         (0.43, 369,  0.439),
    'PaGCN':      (0.20, 401,  0.383),
    'MATE':       (1.26, 1569, 0.404),
    'NESS':       (1.31, 2355, 0.454),
}

fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(111, projection='3d')
for m, (mem, t, f1) in data.items():
    is_ness = (m == 'NESS')
    ax.scatter(mem, t, f1, s=160 if is_ness else 90,
               color='#d62728' if is_ness else '#1f77b4',
               edgecolor='black', linewidth=0.5, depthshade=True)
    # drop line to the F1=floor plane, to anchor depth perception
    ax.plot([mem, mem], [t, t], [0.31, f1], color='gray', lw=0.6, alpha=0.5)
    ax.text(mem, t, f1 + 0.004, m, fontsize=9,
            weight='bold' if is_ness else 'normal')

ax.set_xlabel('Peak GPU mem (GB)', fontsize=11, labelpad=8)
ax.set_ylabel('Train time (s)', fontsize=11, labelpad=8)
ax.set_zlabel('Test macro-F1', fontsize=11, labelpad=6)
ax.set_zlim(0.31, 0.47)
ax.set_title('Accuracy vs. cost on ogbn-arxiv', fontsize=13)
ax.view_init(elev=18, azim=-60)   # viewing angle
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'results/e6_scalability_3d.{ext}', dpi=200)
print('saved e6_scalability_3d')
