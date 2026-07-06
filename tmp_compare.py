from jobs.scoring_engine import get_engine

events = [
    {'semantic_risk': 0.85, 'velocity_1h': 12, 'proxy_score': 1.0, 'amount_deviation': 2.1},
    {'semantic_risk': 0.2, 'velocity_1h': 3, 'proxy_score': 0.0, 'amount_deviation': 0.3},
    {'semantic_risk': 0.6, 'velocity_1h': 25, 'proxy_score': 0.5, 'amount_deviation': 1.5},
]
engines = ['gmdh', 'bedrock_mock', 'ollama']
print('=' * 80)
print('ENGINE COMPARISON (same inputs)')
print('=' * 80)
print(f"| {'#':>2} | {'GMDH':>12} | {'BEDROCK_MOCK':>12} | {'OLLAMA':>12} | {'AGREE':>5} |")
print('-' * 80)
for i, tx in enumerate(events):
    results = {name: get_engine(name).score_transactions([tx], model_path='data/fraud_model_coeffs.json')[0] for name in engines}
    decisions = [r['decision'] for r in results.values() if r['decision'] != 'ERROR']
    agree = 'YES' if len(set(decisions)) == 1 else 'NO'
    gmdh_str = f"{results['gmdh']['score']:.3f} {results['gmdh']['decision']}"
    mock_str = f"{results['bedrock_mock']['score']:.3f} {results['bedrock_mock']['decision']}"
    ollama_str = f"{results['ollama']['score']:.3f} {results['ollama']['decision']}"
    print(f"| {i + 1:>2} | {gmdh_str:>12} | {mock_str:>12} | {ollama_str:>12} | {agree:>5} |")
print('=' * 80)
