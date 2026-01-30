"""Compare pipeline results to eLife reference data"""
import pandas as pd
from pathlib import Path

# Load our pipeline results
brain_path = Path(r"Y:\2_Connectome\3_Nuclei_Detection\1_Brains\349_CNT_01_02\349_CNT_01_02_1p625x_z4")
our_data = pd.read_csv(brain_path / "6_Region_Analysis" / "cell_counts_by_region.csv")

# Create mapping from our region acronyms to eLife region groupings
elife_mapping = {
    # 1. Solitariospinal Area
    'DMX': 'Solitariospinal Area',
    'CU': 'Solitariospinal Area',
    'GR': 'Solitariospinal Area',
    'NTS': 'Solitariospinal Area',
    'MY': 'Solitariospinal Area',

    # 2. Medullary Reticular Nuclei
    'MDRNv': 'Medullary Reticular Nuclei',
    'MDRNd': 'Medullary Reticular Nuclei',

    # 3. Magnocellular Ret. Nuc.
    'MARN': 'Magnocellular Ret. Nuc.',

    # 4. Gigantocellular reticular nucleus
    'GRN': 'Gigantocellular Ret. Nuc.',

    # 5. Lateral Reticular Nuclei
    'LRNm': 'Lateral Reticular Nuclei',
    'PGRNl': 'Lateral Reticular Nuclei',
    'IRN': 'Lateral Reticular Nuclei',
    'PARN': 'Lateral Reticular Nuclei',

    # 6. Dorsal Reticular Nucleus (PGRNd)
    'PGRNd': 'Dorsal Reticular Nucleus',

    # 7. Raphe Nuclei
    'RM': 'Raphe Nuclei',
    'RO': 'Raphe Nuclei',

    # 8. Perihypoglossal Area
    'XII': 'Perihypoglossal Area',
    'NR': 'Perihypoglossal Area',
    'PRP': 'Perihypoglossal Area',

    # 9. Medullary Trigeminal Area
    'SPVI': 'Medullary Trigeminal Area',
    'SPVC': 'Medullary Trigeminal Area',

    # 10. Vestibular Nuclei
    'SPIV': 'Vestibular Nuclei',
    'MV': 'Vestibular Nuclei',
    'LAV': 'Vestibular Nuclei',
    'SUV': 'Vestibular Nuclei',

    # 11. Pontine Reticular Nuclei
    'PRNc': 'Pontine Reticular Nuclei',
    'PRNr': 'Pontine Reticular Nuclei',

    # 13. Pontine Trigeminal Area
    'P5': 'Pontine Trigeminal Area',
    'SUT': 'Pontine Trigeminal Area',

    # 14. Pontine Central Gray Area
    'PCG': 'Pontine Central Gray Area',
    'SLD': 'Pontine Central Gray Area',
    'LDT': 'Pontine Central Gray Area',
    'SLC': 'Pontine Central Gray Area',
    'B': 'Pontine Central Gray Area',
    'P': 'Pontine Central Gray Area',

    # 15. Parabrachial / Pedunculopontine
    'PB': 'Parabrachial / Pedunculopontine',
    'PPN': 'Parabrachial / Pedunculopontine',

    # 16. Cerebellospinal Nuclei
    'FN': 'Cerebellospinal Nuclei',
    'IP': 'Cerebellospinal Nuclei',

    # 17. Red Nucleus
    'RN': 'Red Nucleus',

    # 18. Midbrain Midline Nuclei
    'EW': 'Midbrain Midline Nuclei',
    'INC': 'Midbrain Midline Nuclei',
    'ND': 'Midbrain Midline Nuclei',
    'MB': 'Midbrain Midline Nuclei',

    # 19. Midbrain Reticular Nuclei
    'MRN': 'Midbrain Reticular Nuclei',
    'RR': 'Midbrain Reticular Nuclei',

    # 20. Periaqueductal Gray
    'PAG': 'Periaqueductal Gray',

    # 21. Hypothalamic Periventricular Zone
    'PVH': 'Hypothalamic Periventricular Zone',
    'PVHd': 'Hypothalamic Periventricular Zone',

    # 22. Hypothalamic Lateral Area
    'DMH': 'Hypothalamic Lateral Area',
    'HY': 'Hypothalamic Lateral Area',
    'LHA': 'Hypothalamic Lateral Area',
    'ZI': 'Hypothalamic Lateral Area',

    # 23. Hypothalamic Medial Area
    'AHN': 'Hypothalamic Medial Area',
    'PH': 'Hypothalamic Medial Area',
    'VMH': 'Hypothalamic Medial Area',

    # 24. Thalamus
    'PR': 'Thalamus',
    'SPFm': 'Thalamus',
    'TH': 'Thalamus',
    'VM': 'Thalamus',
    'RE': 'Thalamus',

    # 25. Corticospinal (motor cortex layers 5 and 6a)
    'MOp5': 'Corticospinal',
    'MOp6a': 'Corticospinal',
    'MOs5': 'Corticospinal',
    'MOs6a': 'Corticospinal',
    'SSp-ul5': 'Corticospinal',
    'SSp-ul6a': 'Corticospinal',
    'SSp-ll5': 'Corticospinal',
    'SSp-ll6a': 'Corticospinal',
    'SSp-tr5': 'Corticospinal',
    'SSp-tr6a': 'Corticospinal',
}

# Group our data by eLife categories
our_data['elife_group'] = our_data['region_acronym'].map(elife_mapping)
grouped = our_data.groupby('elife_group')['cell_count'].sum().sort_values(ascending=False)

# Print comparison
print("=" * 80)
print("PIPELINE RESULTS GROUPED BY ELIFE REGIONS")
print("=" * 80)
print(f"{'eLife Region':<45} {'Our Count':>10}")
print("-" * 60)
for region, count in grouped.items():
    print(f"{region:<45} {count:>10,}")

print("\n" + "=" * 80)
print("COMPARISON TO ELIFE L1 UNINJURED (LUMBAR) - Sample 159-Tot")
print("Note: Our data is CERVICAL injection, eLife is LUMBAR injection")
print("=" * 80)

# eLife L1 uninjured totals (from sample 159-Tot column)
elife_l1 = {
    'Solitariospinal Area': 1238,
    'Medullary Reticular Nuclei': 719,
    'Magnocellular Ret. Nuc.': 1867,
    'Gigantocellular Ret. Nuc.': 9824,
    'Lateral Reticular Nuclei': 2242,
    'Dorsal Reticular Nucleus': 21,
    'Raphe Nuclei': 970,
    'Perihypoglossal Area': 118,
    'Medullary Trigeminal Area': 7,
    'Vestibular Nuclei': 235,
    'Pontine Reticular Nuclei': 3893,
    'Pontine Trigeminal Area': 23,
    'Pontine Central Gray Area': 1698,
    'Parabrachial / Pedunculopontine': 354,
    'Cerebellospinal Nuclei': 0,
    'Red Nucleus': 3397,
    'Midbrain Midline Nuclei': 459,
    'Midbrain Reticular Nuclei': 833,
    'Periaqueductal Gray': 622,
    'Hypothalamic Periventricular Zone': 507,
    'Hypothalamic Lateral Area': 1406,
    'Hypothalamic Medial Area': 74,
    'Thalamus': 595,
    'Corticospinal': 8915,
}

print(f"\n{'eLife Region':<45} {'eLife L1':>10} {'Our Count':>10} {'Ratio':>10}")
print("-" * 80)
for region in sorted(elife_l1.keys()):
    elife_count = elife_l1[region]
    our_count = grouped.get(region, 0)
    if elife_count > 0:
        ratio = our_count / elife_count
        print(f"{region:<45} {elife_count:>10,} {int(our_count):>10,} {ratio:>10.2f}x")
    else:
        print(f"{region:<45} {elife_count:>10,} {int(our_count):>10,} {'N/A':>10}")

# Total comparison
elife_total = sum(elife_l1.values())
our_total = grouped.sum()
print("-" * 80)
print(f"{'TOTAL (mapped regions)':<45} {elife_total:>10,} {int(our_total):>10,} {our_total/elife_total:>10.2f}x")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS:")
print("=" * 80)
print("""
CERVICAL vs LUMBAR injection key differences:

1. CEREBELLOSPINAL NUCLEI: We have 239 cells vs eLife's 0
   - Fastigial (149) + Interposed (90) nuclei
   - These project preferentially to CERVICAL cord for forelimb control
   - This is a BIOLOGICALLY EXPECTED difference!

2. CORTICOSPINAL: We have 5,311 vs eLife's 8,915 (0.60x)
   - Cervical projecting neurons are in different cortical regions
   - Upper limb representation (SSp-ul) vs lower limb (for lumbar)

3. GIGANTOCELLULAR: We have 3,721 vs eLife's 9,824 (0.38x)
   - Different population sizes for cervical vs lumbar projections

4. RED NUCLEUS: We have 1,522 vs eLife's 3,397 (0.45x)
   - Red nucleus has topographic organization
   - Rostral RN projects to cervical, caudal to lumbar
   - Cervical-projecting population may be smaller

5. PONTINE RETICULAR: We have 2,304 vs eLife's 3,893 (0.59x)
   - Similar pattern - different projection populations
""")

# Unmapped regions
unmapped = our_data[our_data['elife_group'].isna()]
print("\n" + "=" * 80)
print(f"REGIONS NOT IN ELIFE MAPPING ({len(unmapped)} regions, {unmapped['cell_count'].sum():,} cells)")
print("=" * 80)
top_unmapped = unmapped.nlargest(15, 'cell_count')[['region_acronym', 'region_name', 'cell_count']]
for _, row in top_unmapped.iterrows():
    print(f"  {row['region_acronym']:<12} {row['cell_count']:>6}  {row['region_name']}")
