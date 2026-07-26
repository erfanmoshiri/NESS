"""Figure A: probe class-relevance of each objective target vs. its downstream lift."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# probe macro-F1 (target class-relevance) from probe_objectives.py
probe = {'hist': 0.4745, 'centroid': 0.3557, 'recon': 0.2185, 'anchor': 0.1406, 'stats': 0.0856}
# downstream Delta F1 vs no_ssl, from E3 (locked config @0.8)
dF1   = {'hist': 0.020, 'centroid': 0.007, 'recon': 0.017, 'anchor': 0.004, 'stats': -0.003}

plt.figure(figsize=(5.2, 4.2))
for k in probe:
    plt.scatter(probe[k], dF1[k], s=60)
    plt.annotate(k, (probe[k], dF1[k]), textcoords='offset points', xytext=(6, 4), fontsize=9)
plt.axhline(0, color='gray', lw=0.8, ls='--')
plt.xlim(0.05, 0.52)
plt.ylim(-0.005, 0.024)
plt.xlabel('Target class-relevance (probe macro-F1)')
plt.ylabel(r'Downstream lift $\Delta$F1 (vs no-SSL)')
plt.title('Objective target class-relevance vs. downstream lift')
plt.grid(alpha=0.25)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(f'results/figA_probe_vs_lift.{ext}', dpi=200)
print('saved results/figA_probe_vs_lift.png/.pdf')
