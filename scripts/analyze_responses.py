#!/usr/bin/env python3
"""
Analyze survey responses by deshuffling the method positions and creating
a distribution chart for each method.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Method mapping from study.html
methods = ['human', 'vla', 'diffusion', 'risk']

# METHOD_SHUFFLE_ORDER from study.html lines 319-335
# Each inner array contains indices 0-3 corresponding to methods array
# Positions: 0=Upper Left, 1=Upper Right, 2=Lower Left, 3=Lower Right
METHOD_SHUFFLE_ORDER = [
    [2, 0, 3, 1],  # Trial 1 (Video 1)
    [3, 2, 1, 0],  # Trial 3 (Video 2)
    [2, 3, 0, 1],  # Trial 5 (Video 3)
    [1, 0, 3, 2],  # Trial 6 (Video 4)
    [0, 2, 1, 3],  # Trial 8 (Video 5)
    [2, 1, 0, 3],  # Trial 11 (Video 6)
    [0, 3, 1, 2],  # Trial 12 (Video 7)
    [0, 2, 3, 1],  # Trial 16 (Video 8)
    [3, 2, 0, 1],  # Trial 17 (Video 9)
    [1, 0, 2, 3],  # Trial 18 (Video 10)
    [2, 3, 1, 0],  # Trial 19 (Video 11)
    [2, 0, 1, 3],  # Trial 23 (Video 12)
    [0, 3, 2, 1],  # Trial 24 (Video 13)
    [1, 3, 0, 2],  # Trial 25 (Video 14)
    [2, 0, 3, 1],  # Trial 27 (Video 15)
    [0, 1, 2, 3]   # Trial 28 (Video 16)
]

# Position labels in the CSV
position_labels = ['Upper Left', 'Upper Right', 'Lower Left', 'Lower Right']

def extract_rating(rating_str):
    """Extract numeric rating from string like '1 (Safest)' or '4 (Riskiest)'"""
    if pd.isna(rating_str):
        return None
    # Extract the first digit
    import re
    match = re.search(r'(\d+)', str(rating_str))
    if match:
        return int(match.group(1))
    return None

def analyze_responses(csv_path):
    """Read CSV and deshuffle responses to get method ratings"""
    df = pd.read_csv(csv_path)

    # Initialize counts: method -> rating -> count
    method_ratings = {
        'human': {1: 0, 2: 0, 3: 0, 4: 0},
        'vla': {1: 0, 2: 0, 3: 0, 4: 0},
        'diffusion': {1: 0, 2: 0, 3: 0, 4: 0},
        'risk': {1: 0, 2: 0, 3: 0, 4: 0}
    }

    # Process each response (row)
    for idx, row in df.iterrows():
        # Skip header row if it exists
        if idx == 0 and 'Timestamp' in str(row.iloc[0]):
            continue

        # Process each trial (Video 1-16)
        for trial_idx in range(16):
            video_num = trial_idx + 1
            shuffle_order = METHOD_SHUFFLE_ORDER[trial_idx]

            # Get ratings for each position in this trial
            position_ratings = []
            for pos_idx, pos_label in enumerate(position_labels):
                col_name = f'Video {video_num}: Safest to Riskiest [{pos_label}]'
                # Handle slight variations in column names
                if col_name not in df.columns:
                    # Try with extra space
                    col_name = f'Video {video_num}: Safest to Riskiest [{pos_label} ]'
                if col_name not in df.columns:
                    # Try without space before bracket
                    col_name = f'Video {video_num}: Safest to Riskiest[{pos_label}]'

                if col_name in df.columns:
                    rating = extract_rating(row[col_name])
                    position_ratings.append(rating)
                else:
                    position_ratings.append(None)

            # Map position ratings to method ratings
            for pos_idx, method_idx in enumerate(shuffle_order):
                rating = position_ratings[pos_idx]
                if rating is not None:
                    method_name = methods[method_idx]
                    method_ratings[method_name][rating] += 1

    return method_ratings

def plot_distribution(method_ratings, output_path='rating_distribution.png'):
    """Create a grouped bar chart showing rating distribution per method"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for plotting
    method_names = ['Human\n(Tele-operated)', 'GR00T\n(VLA)', 'Diffusion\nPolicy', 'Risk\n(Ours)']
    ratings = [1, 2, 3, 4]

    # Width of bars and positions
    x = np.arange(len(ratings))
    width = 0.2

    # Colors for each method
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # Plot bars for each method
    for i, (method_key, label) in enumerate(zip(['human', 'vla', 'diffusion', 'risk'], method_names)):
        counts = [method_ratings[method_key][rating] for rating in ratings]
        offset = (i - 1.5) * width
        ax.bar(x + offset, counts, width, label=label, color=colors[i], alpha=0.8)

    # Customize plot
    ax.set_xlabel('Rating (1 = Safest, 4 = Riskiest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Risk Ratings by Method', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['1\n(Safest)', '2', '3', '4\n(Riskiest)'])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (method_key, label) in enumerate(zip(['human', 'vla', 'diffusion', 'risk'], method_names)):
        counts = [method_ratings[method_key][rating] for rating in ratings]
        offset = (i - 1.5) * width
        for j, count in enumerate(counts):
            if count > 0:
                ax.text(j + offset, count + 0.5, str(count),
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

    fig.savefig('scripts/rating_distribution.png', dpi=300, bbox_inches='tight')

    return fig

def print_summary(method_ratings):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("RATING DISTRIBUTION SUMMARY")
    print("="*60)

    for method in ['human', 'vla', 'diffusion', 'risk']:
        label = {
            'human': 'Human (Tele-operated)',
            'vla': 'GR00T (VLA)',
            'diffusion': 'Diffusion Policy',
            'risk': 'Risk (Ours)'
        }[method]

        print(f"\n{label}:")
        total = sum(method_ratings[method].values())
        for rating in [1, 2, 3, 4]:
            count = method_ratings[method][rating]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Rating {rating}: {count:3d} ({pct:5.1f}%)")

        # Calculate average rating
        if total > 0:
            avg = sum(rating * count for rating, count in method_ratings[method].items()) / total
            print(f"  Average: {avg:.2f}")
            print(f"  Total:   {total}")

if __name__ == '__main__':
    csv_path = '/home/admin/Projects/bayesian_risk-website/scripts/form_responses.csv'

    print("Analyzing survey responses...")
    method_ratings = analyze_responses(csv_path)

    print_summary(method_ratings)

    print("\nGenerating visualization...")
    plot_distribution(method_ratings)
