#!/usr/bin/env python3
"""
Create category-level summary from fp_summary.csv
Generates category_summary.csv for paper tables
"""

import csv
from collections import defaultdict

# Virus to category mapping (from user specification)
VIRUS_CATEGORIES = {
    # Large DNA
    'hpv': 'Large DNA',
    'hsv1': 'Large DNA',
    'vzv': 'Large DNA',
    'adenovirus': 'Large DNA',
    'varv': 'Large DNA',

    # Small DNA
    'parvovirus_b19': 'Small DNA',

    # +ssRNA
    'sars_cov2': '+ssRNA',
    'mers': '+ssRNA',
    'oc43': '+ssRNA',
    'hku1': '+ssRNA',
    'nl63': '+ssRNA',
    '229e': '+ssRNA',
    'jev': '+ssRNA',
    'dengue': '+ssRNA',
    'hcv': '+ssRNA',
    'hiv': '+ssRNA',

    # -ssRNA
    'rabies': '-ssRNA',
    'measles': '-ssRNA',
    'mumps': '-ssRNA',
    'influenza': '-ssRNA',

    # dsRNA
    'reovirus': 'dsRNA',

    # Enteric
    'poliovirus': 'Enteric',
    'norovirus': 'Enteric'
}

def main():
    # Read fp_summary.csv and aggregate by category
    category_stats = defaultdict(lambda: {
        'viruses_in_category': 0,
        'viruses_successful': 0,
        'total_verified_hits': 0,
        'total_claimed_hits': 0,
        'viruses_list': []
    })

    with open('fp_summary.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] == 'Evo2-7B':  # Only process 7B results
                virus = row['virus']
                category = VIRUS_CATEGORIES.get(virus, 'Unknown')

                blastn_verified = int(row['blastn_verified'])
                blastp_verified = int(row['blastp_verified'])
                blastn_claims = int(row['blastn_claims'])
                blastp_claims = int(row['blastp_claims'])

                total_verified = blastn_verified + blastp_verified
                total_claims = blastn_claims + blastp_claims

                stats = category_stats[category]
                stats['viruses_in_category'] += 1
                stats['total_verified_hits'] += total_verified
                stats['total_claimed_hits'] += total_claims
                stats['viruses_list'].append(virus)

                if total_verified > 0:
                    stats['viruses_successful'] += 1

    # Write category summary
    with open('category_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'category', 'viruses_in_category', 'viruses_successful',
            'total_verified_hits', 'total_claimed_hits', 'virus_success_rate_pct',
            'hit_verification_rate_pct', 'virus_list'
        ])

        # Sort categories by success rate
        categories = sorted(category_stats.items(),
                          key=lambda x: x[1]['viruses_successful'] / x[1]['viruses_in_category'],
                          reverse=True)

        for category, stats in categories:
            virus_success_rate = (stats['viruses_successful'] / stats['viruses_in_category']) * 100
            hit_verification_rate = (stats['total_verified_hits'] / max(stats['total_claimed_hits'], 1)) * 100
            virus_list = '; '.join(stats['viruses_list'])

            writer.writerow([
                category,
                stats['viruses_in_category'],
                stats['viruses_successful'],
                stats['total_verified_hits'],
                stats['total_claimed_hits'],
                f"{virus_success_rate:.1f}",
                f"{hit_verification_rate:.1f}",
                virus_list
            ])

    print("Category summary created: category_summary.csv")
    print("\nBreakdown:")
    for category, stats in categories:
        virus_success_rate = (stats['viruses_successful'] / stats['viruses_in_category']) * 100
        print(f"{category:12}: {stats['viruses_successful']}/{stats['viruses_in_category']} viruses ({virus_success_rate:.1f}%), {stats['total_verified_hits']} verified hits")

if __name__ == "__main__":
    main()