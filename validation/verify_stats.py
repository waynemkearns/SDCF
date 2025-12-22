#!/usr/bin/env python3
"""Verify statistics from JSON for Paper 2 corrections."""

import json
import statistics

with open('bronze_retrospective_results.json') as f:
    data = json.load(f)

bprs = [d['b_prs'] for d in data]
bfi = [d['b_fi'] for d in data]
bfv = [d['b_fv'] for d in data]
domains = set(d['domain'] for d in data)

print('='*60)
print('VERIFIED STATISTICS FROM JSON')
print('='*60)

print(f'\nB-PRS:')
print(f'  Mean:  {statistics.mean(bprs):.3f}')
print(f'  SD:    {statistics.stdev(bprs):.3f}')
print(f'  Range: {min(bprs):.3f} - {max(bprs):.3f}')
print(f'  Median: {statistics.median(bprs):.3f}')

print(f'\nB-FI:')
print(f'  Mean:  {statistics.mean(bfi):.3f}')
print(f'  SD:    {statistics.stdev(bfi):.3f}')

print(f'\nB-FV (Fairness):')
problematic = sum(1 for v in bfv if v > 0.30)
print(f'  Problematic (>0.30): {problematic}/10 = {problematic*10}%')

print(f'\nDomains ({len(domains)}):')
for d in sorted(domains):
    count = sum(1 for x in data if x['domain'] == d)
    print(f'  - {d} ({count} datasets)')

print(f'\nMethod Comparison (Adult dataset - controlled experiment):')
tvae = next(d for d in data if 'TVAE' in d['dataset_name'])
ctgan = next(d for d in data if 'CTGAN' in d['dataset_name'])
gc = next(d for d in data if 'GaussianCopula' in d['dataset_name'])
print(f'  TVAE B-PRS:           {tvae["b_prs"]:.3f}')
print(f'  CTGAN B-PRS:          {ctgan["b_prs"]:.3f}')
print(f'  GaussianCopula B-PRS: {gc["b_prs"]:.3f}')
diff = ctgan['b_prs'] - tvae['b_prs']
pct = (diff / ctgan['b_prs']) * 100
print(f'  TVAE vs CTGAN diff:   {diff:.3f} ({pct:.1f}% lower privacy risk)')

print(f'\nClassifications:')
low = sum(1 for d in data if d['prs_level'] == 'Low')
mod = sum(1 for d in data if d['prs_level'] == 'Moderate')
high = sum(1 for d in data if d['prs_level'] == 'High')
crit = sum(1 for d in data if d['prs_level'] == 'Critical')
print(f'  Low: {low}, Moderate: {mod}, High: {high}, Critical: {crit}')
print(f'  Distribution: {low*10}-{mod*10}-{(high+crit)*10} (Low-Mod-High+Critical)')

print(f'\nConformance:')
sdcf_a = sum(1 for d in data if d['conformance'] == 'SDCF-A-Bronze')
sdcf_p = sum(1 for d in data if d['conformance'] == 'SDCF-P-Bronze')
sdcf_r = sum(1 for d in data if d['conformance'] == 'SDCF-R-Bronze')
print(f'  SDCF-A (Acceptable): {sdcf_a}')
print(f'  SDCF-P (Provisional): {sdcf_p}')
print(f'  SDCF-R (Rejected): {sdcf_r}')

print('\n' + '='*60)
print('TABLE 1 DATA (for Paper 2)')
print('='*60)
print(f'\n{"ID":<4} {"Dataset":<32} {"Domain":<12} {"Records":>8} {"B-PRS":>7} {"B-FI":>6} {"B-FV":>6} {"Class":<10}')
print('-'*95)
for d in data:
    print(f'{d["dataset_id"]:<4} {d["dataset_name"]:<32} {d["domain"]:<12} {d["n_records"]:>8,} {d["b_prs"]:>7.3f} {d["b_fi"]:>6.3f} {d["b_fv"]:>6.3f} {d["conformance"]:<10}')



