"""E6 scalability scatter: peak GPU memory (x) vs Test F1 (y), one point per method.
Shows NESS reaches top accuracy at memory comparable to other GNNs.
Source: results/e6_scalability_20260725_091754/summary.txt"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# model: (peak GPU GB from E6 run, test F1 from E1 @0.4 main comparison, train time s)
# F1 taken at the main-comparison rate (0.4) so the accuracy axis matches Table~\ref{tab:main}.
data = {
    'NeighAggre': (0.48, 0.325, 46),
    'KNN':        (0.26, 0.335, 49),
    'GraphSAGE':  (0.14, 0.372, 359),
    'FP':         (0.43, 0.439, 369),
    'PaGCN':      (0.20, 0.383, 401),
    'MATE':       (1.26, 0.404, 1569),
    'NESS':       (1.31, 0.454, 2355),
}

plt.figure(figsize=(6.2, 4.4))
for m, (mem, f1, t) in data.items():
    is_ness = (m == 'NESS')
    plt.scatter(mem, f1, s=140 if is_ness else 90,
                color='#d62728' if is_ness else '#1f77b4', zorder=3,
                edgecolor='black', linewidth=0.5)
    # NESS and MATE share high memory; offset labels so they don't collide
    if m == 'NESS':
        off = (-6, 9)
    elif m == 'MATE':
        off = (-8, -16)
    else:
        off = (8, 3)
    plt.annotate(m, (mem, f1), textcoords='offset points', xytext=off,
                 fontsize=10, weight='bold' if is_ness else 'normal')
plt.xlim(0.05, 1.55)
plt.ylim(0.31, 0.475)
plt.xlabel('Peak GPU memory (GB)', fontsize=12)
plt.ylabel('Test macro-F1', fontsize=12)
plt.title('Accuracy vs. memory on ogbn-arxiv', fontsize=12)
plt.grid(alpha=0.25)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(f'results/e6_scalability.{ext}', dpi=200)
print('saved e6_scalability')
