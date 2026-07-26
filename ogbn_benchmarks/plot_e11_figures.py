"""
E11 figures.
  Fig 1 (HP sensitivity): 4 line panels (w_cls, dropout, num_layers, lr) vs Test F1,
        from the one-at-a-time sweep (base config, base F1=0.3924 at each knob's default).
        Shows the model peaks near / is stable around the chosen defaults.
  Fig 2 (design choices): grouped bar chart (encoder type, prefill strategy), locked config.
Outputs: results/e11_hp_sensitivity.{png,pdf}, results/e11_design_choices.{png,pdf}
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- Fig 1: HP sensitivity (old base sweep, base test-F1 = 0.3924 at defaults) ----------
BASE = 0.3924
hp = {
    'w_cls':      ([0.5, 1, 2, 3, 5],        [0.3898, BASE, 0.4016, 0.4157, 0.4200], 3),
    'dropout':    ([0.3, 0.5, 0.7],          [0.4044, BASE, 0.3629],                 0.3),
    'num_layers': ([1, 2, 3],                [0.3520, BASE, 0.4049],                 3),
    'lr':         ([5e-4, 1e-3, 5e-3],       [0.3938, BASE, 0.3657],                 1e-3),
}
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (name, (xs, ys, default)) in zip(axes, hp.items()):
    ax.plot(range(len(xs)), ys, 'o-', color='#1f77b4', lw=2)
    di = xs.index(default)
    ax.plot(di, ys[di], 'o', color='#d62728', ms=11, label='chosen')
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_title(name, fontsize=13)
    ax.set_xlabel(name, fontsize=12)
    ax.grid(alpha=0.25)
axes[0].set_ylabel('Test macro-F1', fontsize=12)
axes[0].legend(frameon=False, fontsize=11)
fig.suptitle('Hyperparameter sensitivity (one knob varied at a time; red = chosen default)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ('png', 'pdf'):
    fig.savefig(f'results/e11_hp_sensitivity.{ext}', dpi=200)
print('saved e11_hp_sensitivity')

# ---------- Fig 2: design choices (locked config) ----------
fig2, ax = plt.subplots(figsize=(7, 4.0))
groups = [
    ('Encoder', ['GCN', 'GAT', 'SAGE'],  [0.354, 0.388, 0.401], 2),   # SAGE chosen
    ('Prefill', ['zero', 'mean', 'FP'],  [0.3497, 0.4205, 0.4466], 2), # FP chosen
]
xpos, labels = [], []
x = 0
group_centers = []
for gname, opts, vals, chosen in groups:
    xs_g = []
    for j, (o, v) in enumerate(zip(opts, vals)):
        c = '#d62728' if j == chosen else '#1f77b4'
        ax.bar(x, v, color=c, width=0.8)
        ax.text(x, v + 0.005, f'{v:.3f}', ha='center', fontsize=11)
        labels.append(o); xpos.append(x); xs_g.append(x); x += 1
    group_centers.append(sum(xs_g) / len(xs_g))
    x += 1.0  # gap between groups
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=12)
ax.tick_params(axis='y', labelsize=11)
ax.set_ylabel('Test macro-F1', fontsize=13); ax.set_ylim(0.3, 0.48)
# group names BELOW the tick labels (second line under the axis)
for cx, (gname, *_ ) in zip(group_centers, groups):
    ax.annotate(gname, xy=(cx, -0.19), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=13, weight='bold', annotation_clip=False)
ax.set_title('Design choices (red = chosen); 80% missing', fontsize=13)
ax.grid(alpha=0.25, axis='y')
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.24)
for ext in ('png', 'pdf'):
    plt.savefig(f'results/e11_design_choices.{ext}', dpi=200)
print('saved e11_design_choices')
