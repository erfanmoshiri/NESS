"""
E10 Figure 1 — per-objective SSL loss curves.

Shows the mechanism behind the objective ablation (E3): blunt objectives
(centroid, stats, recon) saturate within a few epochs, while useful objectives
(hist, path) keep decreasing throughout training. Each SSL loss has a different
scale, so we plot each curve NORMALIZED to its own epoch-1 value (fraction of the
initial loss remaining) — a flat line means "learned early, nothing left"; a line
that keeps dropping means "keeps teaching".

Source logs: results/e3_ablation_20260723_094037/ssl_<obj>.log (one objective each).

Usage:
    python plot_e10_loss_curves.py
Outputs: results/e10_loss_curves.png (and .pdf)
"""
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E3_DIR = 'results/e3_ablation_20260723_094037'
OUT = 'results/e10_loss_curves'

# (log file, printed loss key, legend label, "blunt" vs "useful")
OBJECTIVES = [
    ('ssl_centroid.log', 'Centroid', 'centroid (feature)', 'blunt'),
    ('ssl_stats.log',    'Stats',    'stats (feature)',    'blunt'),
    ('ssl_recon.log',    'Recon',    'recon (feature)',    'blunt'),
    ('ssl_path.log',     'Path',     'path (structural)',  'useful'),
    ('ssl_hist.log',     'Hist',     'hist (label)',       'useful'),
]

BLUNT_STYLE  = dict(linestyle='--', alpha=0.9)
USEFUL_STYLE = dict(linestyle='-',  linewidth=2.2)


def parse_curve(path, key):
    """Return (epochs, losses) for the given printed loss key."""
    pat = re.compile(r'Epoch (\d+)/\d+.*?%s: ([0-9.]+)' % key)
    ep, val = [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ep.append(int(m.group(1)))
                val.append(float(m.group(2)))
    return ep, val


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    curves = {}
    for fname, key, label, kind in OBJECTIVES:
        p = os.path.join(E3_DIR, fname)
        if not os.path.exists(p):
            print(f'  skip (missing): {fname}'); continue
        ep, val = parse_curve(p, key)
        if not val:
            print(f'  skip (no data): {fname}'); continue
        curves[label] = (ep, val, kind)
        print(f'  {label:22s} start={val[0]:.3f} end={val[-1]:.3f}')

    XCAP = 400
    def cap(ep, val):
        exy = [(e, v) for e, v in zip(ep, val) if e <= XCAP]
        return [e for e, _ in exy], [v for _, v in exy]

    # left panel: blunt (feature) objectives, their own linear scale
    for label, (ep, val, kind) in curves.items():
        if kind == 'blunt':
            cx, cy = cap(ep, val)
            axL.plot(cx, cy, label=label, **BLUNT_STYLE)
    axL.set_title('Feature objectives:\nsaturate within a few epochs')
    axL.set_xlabel('Epoch'); axL.set_ylabel('SSL loss')
    axL.set_ylim(bottom=0); axL.set_xlim(0, XCAP)
    axL.legend(frameon=False, fontsize=9); axL.grid(alpha=0.25)

    # right panel: useful (structural / label) objectives, their own linear scale
    for label, (ep, val, kind) in curves.items():
        if kind == 'useful':
            cx, cy = cap(ep, val)
            axR.plot(cx, cy, label=label, **USEFUL_STYLE)
    axR.set_title('Structural / label objectives:\nkeep decreasing throughout training')
    axR.set_xlabel('Epoch'); axR.set_ylabel('SSL loss')
    axR.set_ylim(bottom=0); axR.set_xlim(0, XCAP)
    axR.legend(frameon=False, fontsize=9); axR.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'{OUT}.{ext}', dpi=200)
    print(f'Saved {OUT}.png / .pdf')


if __name__ == '__main__':
    main()
