"""E12: SSL benefit grows with missingness. Test F1 vs missing rate, no-SSL vs hist+path.
Source: results/e_sslmiss_20260726_144148/summary.txt (locked config)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rates = [0.8, 0.9, 0.95]
no_ssl    = [0.4425, 0.4074, 0.3500]
hist_path = [0.4479, 0.4165, 0.3809]

plt.figure(figsize=(6.0, 4.3))
x = range(len(rates))
plt.plot(x, no_ssl,    'o--', color='#7f7f7f', lw=2, ms=8, label='no SSL')
plt.plot(x, hist_path, 'o-',  color='#d62728', lw=2.4, ms=9, label='NESS (hist + path)')
# shade the widening gap
plt.fill_between(x, no_ssl, hist_path, color='#d62728', alpha=0.12)
for i in x:
    d = hist_path[i] - no_ssl[i]
    plt.annotate(f'+{d:.3f}', (i, hist_path[i]), textcoords='offset points',
                 xytext=(0, 9), ha='center', fontsize=10, color='#d62728', weight='bold')
plt.xticks(list(x), [f'{r:g}' for r in rates], fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel('Missing rate', fontsize=12)
plt.ylabel('Test macro-F1', fontsize=12)
plt.title('Self-supervision helps more as missingness increases', fontsize=12)
plt.legend(frameon=False, fontsize=11, loc='lower left')
plt.grid(alpha=0.25)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(f'results/e12_missingness.{ext}', dpi=200)
print('saved e12_missingness')
