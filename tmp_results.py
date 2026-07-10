import json

def get_m(path):
    d = json.load(open(path))
    if 'metrics' in d:
        m = d['metrics']
        g = d['gate']['overall_pass']
        return m['precision'], m['recall'], m['f1'], m['auc_roc'], g
    g = d.get('gate_status', 'FAIL') == 'PASS'
    return d['precision'], d['recall'], d['f1'], d['auc_roc'], g

files = [
    ('GMDH v3 (Random)',     'data/benchmark_metrics_200k.json'),
    ('GMDH v4 (Self-Org)',   'data/benchmark_metrics_200k_gmdh.json'),
    ('GMDH v5 (Form Search)','data/benchmark_metrics_200k_gmdh_proper.json'),
    ('NN (NumPy)',            'data/benchmark_metrics_nn_200k.json'),
]

print(f"{'Model':<25} {'Prec':>6} {'Recall':>6} {'F1':>6} {'AUC':>6} {'Gate':>5}")
print('-' * 58)
for name, path in files:
    p, r, f, a, g = get_m(path)
    print(f"{name:<25} {p:>6.4f} {r:>6.4f} {f:>6.4f} {a:>6.4f} {'PASS' if g else 'FAIL':>5}")
print('-' * 58)
print(f"{'Thresholds:':<25} {'>=0.50':>6} {'>=0.40':>6} {'>=0.45':>6} {'>=0.78':>6}")
