"""Dry-run the fixed decade allocation against the real source distribution.

Uses the per-decade document counts measured from
Bhawna/ChroniclingAmericaQA (train split) to check that the repaired
allocation in preprocess_expanded.py gives every decade in the configured
1800-1900 range a non-zero chunk budget, and that the budgets sum to at most
MAX_CHUNKS.

This exercises the allocation arithmetic only. It does not embed anything.
"""
MAX_CHUNKS = 50000
CORPUS_START_YEAR = 1800
CORPUS_END_YEAR = 1900

# Measured from the source dataset, train split, 1800 <= yr < 1900.
SOURCE = {
    1800: 3543, 1810: 4016, 1820: 8084, 1830: 18864, 1840: 32154,
    1850: 46996, 1860: 52635, 1870: 58998, 1880: 64114, 1890: 55983,
}

decades = sorted(SOURCE)

# --- allocation, mirroring preprocess_expanded.py ---
target_per_decade = {}
min_per_decade = 3000
remaining = MAX_CHUNKS - min_per_decade * len(decades)
if remaining < 0:
    min_per_decade = MAX_CHUNKS // len(decades)
    remaining = 0
total_late_docs = sum(SOURCE[d] for d in decades if d >= 1830)
for d in decades:
    if d < 1830:
        target_per_decade[d] = min_per_decade
    else:
        prop = SOURCE[d] / total_late_docs if total_late_docs > 0 else 0
        target_per_decade[d] = min_per_decade + int(remaining * prop)

pre_scale = sum(target_per_decade.values())
print(f"decades in range      : {len(decades)}")
print(f"sum before scaling    : {pre_scale}  (MAX_CHUNKS={MAX_CHUNKS}, "
      f"overshoot={pre_scale - MAX_CHUNKS})")

if pre_scale > MAX_CHUNKS:
    scale = MAX_CHUNKS / pre_scale
    for d in decades:
        target_per_decade[d] = max(1, int(target_per_decade[d] * scale))
    print(f"scale factor applied  : {scale:.4f}")

post = sum(target_per_decade.values())
print(f"sum after scaling     : {post}  (fits in MAX_CHUNKS: {post <= MAX_CHUNKS})")
print()
print(f"{'decade':<8}{'source docs':>12}{'chunk budget':>14}{'share':>9}")
print("-" * 43)
for d in decades:
    print(f"{d}s{'':<5}{SOURCE[d]:>12}{target_per_decade[d]:>14}"
          f"{target_per_decade[d]/post*100:>8.1f}%")
print("-" * 43)
print(f"{'total':<8}{sum(SOURCE.values()):>12}{post:>14}")

zero = [d for d in decades if target_per_decade[d] == 0]
sparse = [d for d in decades if target_per_decade[d] / post < 0.05]
print()
print(f"decades with ZERO budget    : {zero or 'none'}")
print(f"decades below 5% of budget  : {sparse or 'none'}")
print(f"sample pool per decade (2x budget, capped at available):")
for d in decades:
    pool = min(SOURCE[d], target_per_decade[d] * 2)
    print(f"  {d}s: {pool} docs sampled of {SOURCE[d]} available")
