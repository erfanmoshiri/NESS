"""E6 scalability bubble: x=peak memory, y=Test F1, bubble size = train time.
F1 from E1 @0.4 (main comparison); memory + time from E6 run."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# model: (peak GPU GB, test F1 @0.4, train time s)
data = {
    'NeighAggre': (0.48, 0.325, 46),
    'KNN':        (0.26, 0.335, 49),
    'GraphSAGE':  (0.14, 0.372, 359),
    'FP':         (0.43, 0.439, 369),
    'PaGCN':      (0.20, 0.383, 401),
    'MATE':       (1.26, 0.404, 1569),
    'NESS':       (1.31, 0.454, 2355),
}

def bubble(t):  # map train time (s) -> marker area
    return 60 + t * 0.55

plt.figure(figsize=(6.6, 4.6))
for m, (mem, f1, t) in data.items():
    is_ness = (m == 'NESS')
    plt.scatter(mem, f1, s=bubble(t),
                color='#d62728' if is_ness else '#4c8fbf',
                alpha=0.85, edgecolor='black', linewidth=0.6, zorder=3)
    off = {'NESS': (10, 8), 'MATE': (8, -16), 'FP': (10, 2), 'PaGCN': (8, 4),
           'GraphSAGE': (8, -14), 'KNN': (8, 4), 'NeighAggre': (8, 4)}.get(m, (8, 4))
    plt.annotate(m, (mem, f1), textcoords='offset points', xytext=off,
                 fontsize=10, weight='bold' if is_ness else 'normal')

plt.xlim(0.0, 1.55)
plt.ylim(0.31, 0.475)
plt.xlabel('Peak GPU memory (GB)', fontsize=12)
plt.ylabel('Test macro-F1', fontsize=12)
plt.title('Accuracy vs. memory on ogbn-arxiv (bubble size = training time)', fontsize=11.5)
plt.grid(alpha=0.25, zorder=0)

# legend for bubble size
for t in (100, 1000, 2000):
    plt.scatter([], [], s=bubble(t), color='gray', alpha=0.5, edgecolor='black',
                linewidth=0.5, label=f'{t}s')
plt.legend(title='train time', frameon=False, labelspacing=1.4,
           fontsize=9, title_fontsize=9, loc='lower right', borderpad=1.0)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(f'results/e6_scalability_bubble.{ext}', dpi=200)
print('saved e6_scalability_bubble')
