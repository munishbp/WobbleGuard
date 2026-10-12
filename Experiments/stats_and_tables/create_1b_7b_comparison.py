#!/usr/bin/env python3
# build the 1B vs 7B comparison table.
# - 1B: manual curation results (from Jailbreak Results Replication(1).csv)
# - 7B: verified results (from category_summary.csv)

import csv

# map the 1B csv's category names onto the standard ones
CATEGORY_MAP = {
    # 1B names -> standard category names
    'positive ssRNA': '+ssRNA',
    'Large DNA': 'Large DNA',
    'negative ssRNA': '-ssRNA',
    'Enteric': 'Enteric',
    'small DNA': 'Small DNA',
    'dsDNA': 'dsRNA'  # reovirus is dsRNA, not dsDNA
}

def analyze_1b_results():
    # roll up the manually curated 1B results by category
    category_stats = {}

    with open('Jailbreak Results Replication(1).csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('Viruses '):  # skip empty rows
                continue

            virus = row['Viruses '].strip()
            virus_type = row['Virus Type'].strip()
            success = int(row['Success']) if row['Success'].strip() else 0

            category = CATEGORY_MAP.get(virus_type, virus_type)

            if category not in category_stats:
                category_stats[category] = {
                    'viruses_total': 0,
                    'viruses_successful': 0,
                    'successful_viruses': []
                }

            category_stats[category]['viruses_total'] += 1
            if success:
                category_stats[category]['viruses_successful'] += 1
                category_stats[category]['successful_viruses'].append(virus)

    return category_stats

def load_7b_results():
    # read the verified 7B numbers from category_summary.csv
    category_7b = {}

    with open('category_summary.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row['category']
            category_7b[category] = {
                'viruses_total': int(row['viruses_in_category']),
                'viruses_successful': int(row['viruses_successful']),
                'total_verified_hits': int(row['total_verified_hits'])
            }

    return category_7b

def main():
    print("Analyzing 1B manual results...")
    results_1b = analyze_1b_results()

    print("Loading 7B verified results...")
    results_7b = load_7b_results()

    # build the comparison table
    with open('1b_vs_7b_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'category',
            '1b_viruses_total', '1b_viruses_successful', '1b_success_rate_pct',
            '7b_viruses_total', '7b_viruses_successful', '7b_success_rate_pct', '7b_verified_hits',
            'model_comparison'
        ])

        # union of categories from both sides
        all_categories = set(results_1b.keys()) | set(results_7b.keys())

        for category in sorted(all_categories):
            # 1B stats, zeroed out if the category is missing
            stats_1b = results_1b.get(category, {'viruses_total': 0, 'viruses_successful': 0})
            success_rate_1b = (stats_1b['viruses_successful'] / max(stats_1b['viruses_total'], 1)) * 100

            # 7B stats, same deal
            stats_7b = results_7b.get(category, {'viruses_total': 0, 'viruses_successful': 0, 'total_verified_hits': 0})
            success_rate_7b = (stats_7b['viruses_successful'] / max(stats_7b['viruses_total'], 1)) * 100

            # who wins this category
            if success_rate_7b > success_rate_1b:
                comparison = f"7B better (+{success_rate_7b - success_rate_1b:.1f}%)"
            elif success_rate_1b > success_rate_7b:
                comparison = f"1B better (+{success_rate_1b - success_rate_7b:.1f}%)"
            else:
                comparison = "Equal"

            writer.writerow([
                category,
                stats_1b['viruses_total'], stats_1b['viruses_successful'], f"{success_rate_1b:.1f}",
                stats_7b['viruses_total'], stats_7b['viruses_successful'], f"{success_rate_7b:.1f}", stats_7b['total_verified_hits'],
                comparison
            ])

    print("Comparison table created: 1b_vs_7b_comparison.csv")
    print("\nCategory-by-category breakdown:")

    for category in sorted(all_categories):
        stats_1b = results_1b.get(category, {'viruses_total': 0, 'viruses_successful': 0})
        stats_7b = results_7b.get(category, {'viruses_total': 0, 'viruses_successful': 0, 'total_verified_hits': 0})

        rate_1b = (stats_1b['viruses_successful'] / max(stats_1b['viruses_total'], 1)) * 100
        rate_7b = (stats_7b['viruses_successful'] / max(stats_7b['viruses_total'], 1)) * 100

        print(f"{category:12}: 1B {stats_1b['viruses_successful']}/{stats_1b['viruses_total']} ({rate_1b:.1f}%) | 7B {stats_7b['viruses_successful']}/{stats_7b['viruses_total']} ({rate_7b:.1f}%)")

        # list which viruses succeeded for 1B
        if category in results_1b and results_1b[category]['successful_viruses']:
            print(f"             1B successes: {', '.join(results_1b[category]['successful_viruses'])}")

if __name__ == "__main__":
    main()
