#!/usr/bin/env python3
"""
Analyze risk value survey responses and create distribution charts for
Shelf and Kitchen environments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def extract_rating(rating_str):
    """Extract numeric rating from string like '1 (Safest)' or '5 (Most Risky)'"""
    if pd.isna(rating_str):
        return None
    # Extract the first digit
    match = re.search(r'(\d+)', str(rating_str))
    if match:
        return int(match.group(1))
    return None

def analyze_responses(csv_path):
    """Read CSV and extract location ratings"""
    df = pd.read_csv(csv_path)

    # Shelf environment locations
    shelf_locations = ['Shelf 1', 'Shelf 2', 'Shelf 3', 'Shelf 4', 'Shelf 5']

    # Kitchen environment locations
    kitchen_locations = [
        'Kitchen to Coffee Machine',
        'Kitchen to Fridge',
        'Kitchen to Laptop',
        'Kitchen to Microwave',
        'Kitchen to Monitor'
    ]

    # Initialize counts: location -> rating -> count
    shelf_ratings = {loc: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for loc in shelf_locations}
    kitchen_ratings = {loc: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for loc in kitchen_locations}

    # Process each response (row)
    for idx, row in df.iterrows():
        # Skip header row if it exists
        if idx == 0 and 'Timestamp' in str(row.iloc[0]):
            continue

        # Process Shelf environment (columns 1-5, after Timestamp)
        for i, location in enumerate(shelf_locations):
            col_name = f'Shelf Environment while holding a Solo Cup, least to most risky [{location}]'
            if col_name in df.columns:
                rating = extract_rating(row[col_name])
                if rating is not None:
                    shelf_ratings[location][rating] += 1

        # Process Kitchen environment (columns 6-10)
        for i, location in enumerate(kitchen_locations):
            col_name = f'Kitchen Environment while holding a Solo Cup, least to most risky [{location}]'
            if col_name in df.columns:
                rating = extract_rating(row[col_name])
                if rating is not None:
                    kitchen_ratings[location][rating] += 1

    return shelf_ratings, kitchen_ratings

def plot_environment_distribution(location_ratings, title, output_path):
    """Create a grouped bar chart showing rating distribution per location"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for plotting
    locations = list(location_ratings.keys())
    ratings = [1, 2, 3, 4, 5]

    # Width of bars and positions
    x = np.arange(len(locations))
    width = 0.15

    # Colors for each rating
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    rating_labels = ['1 (Least Risky)', '2', '3', '4', '5 (Most Risky)']

    # Plot bars for each rating
    for i, (rating, color, label) in enumerate(zip(ratings, colors, rating_labels)):
        counts = [location_ratings[location][rating] for location in locations]
        offset = (i - 2) * width
        ax.bar(x + offset, counts, width, label=label, color=color, alpha=0.8)

    # Customize plot
    ax.set_xlabel('Location', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    # Shorten labels for better fit
    location_labels = [loc.replace('Kitchen to ', '') if 'Kitchen to' in loc else loc for loc in locations]
    ax.set_xticklabels(location_labels, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, rating in enumerate(ratings):
        counts = [location_ratings[location][rating] for location in locations]
        offset = (i - 2) * width
        for j, count in enumerate(counts):
            if count > 0:
                ax.text(j + offset, count + 0.2, str(count),
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

    return fig

def print_summary(location_ratings, environment_name):
    """Print summary statistics"""
    print(f"\n{'='*70}")
    print(f"{environment_name.upper()} RATING DISTRIBUTION SUMMARY")
    print('='*70)

    for location in location_ratings.keys():
        print(f"\n{location}:")
        total = sum(location_ratings[location].values())

        if total > 0:
            for rating in [1, 2, 3, 4, 5]:
                count = location_ratings[location][rating]
                pct = (count / total * 100) if total > 0 else 0
                print(f"  Rating {rating}: {count:3d} ({pct:5.1f}%)")

            # Calculate average rating
            avg = sum(rating * count for rating, count in location_ratings[location].items()) / total
            print(f"  Average: {avg:.2f}")
            print(f"  Total:   {total}")

if __name__ == '__main__':
    csv_path = '/home/admin/Projects/bayesian_risk-website/scripts/risk_value_responses.csv'

    print("Analyzing risk value survey responses...")
    shelf_ratings, kitchen_ratings = analyze_responses(csv_path)

    # Print summaries
    print_summary(shelf_ratings, "Shelf Environment")
    print_summary(kitchen_ratings, "Kitchen Environment")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_environment_distribution(
        shelf_ratings,
        'Shelf Environment - Risk Rating Distribution',
        'shelf_risk_distribution.png'
    )
    plot_environment_distribution(
        kitchen_ratings,
        'Kitchen Environment - Risk Rating Distribution',
        'kitchen_risk_distribution.png'
    )

    print("\nAnalysis complete!")
