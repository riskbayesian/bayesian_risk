#!/usr/bin/env python3
"""
Analyze object pair pairwise comparison survey and rank pairs by risk using Bradley-Terry model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

# Question structure from the Google Form
QUESTIONS = [
    ("Laptop vs Scissors", "Alarm Clock vs Cup"),
    ("Laptop vs Paint", "Laptop vs Knife"),
    ("Flashlight vs Soda", "Dish Soap vs Laptop"),
    ("Aluminum Foil vs Bucket", "Business Cards vs First Aid Kit"),
    ("Laptop vs Scissors", "Flashlight vs Soda Can"),
    ("Alarm Clock vs Cup", "Dish Soap vs Laptop"),
    ("Laptop vs Paint Can", "Monitor vs Spray Bottle"),
    ("Laptop vs Knife", "Flashlight vs Soda Can"),
    ("Laptop vs Scissors", "Laptop vs Solo Cup"),
    ("Laptop vs Knife", "Laptop vs Solo Cup"),
    ("Laptop vs Scissors", "Aluminum Foil vs Bucket"),
    ("Laptop vs Knife", "Dish Soap vs Sink"),
    ("Flashlight vs Soda", "Laptop vs Solo Cup"),
    ("Flashlight vs Soda", "Aluminum Foil vs Bucket"),
    ("Laptop vs Solo Cup", "Aluminum Foil vs Bucket"),
]

def normalize_pair_name(pair_name):
    """Normalize pair names to handle variations"""
    pair_name = pair_name.strip()
    pair_name = pair_name.replace(' Can', '').replace('Paint Can', 'Paint')
    pair_name = pair_name.replace('Soda Can', 'Soda')
    return pair_name

def analyze_pairwise_comparisons(csv_path):
    """Read CSV and extract pairwise comparison data"""
    df = pd.read_csv(csv_path)

    # Build comparison matrix
    # comparisons[i][j] = number of times pair i was chosen over pair j
    pair_to_idx = {}
    idx_to_pair = {}
    comparisons = defaultdict(lambda: defaultdict(int))

    # Process each response (skip first row which is header)
    for resp_idx, row in df.iterrows():
        if resp_idx == 0:  # Skip header
            continue

        # Process each question (columns 1-15, skipping timestamp)
        for q_idx in range(15):
            if q_idx >= len(QUESTIONS):
                break

            pair_a, pair_b = QUESTIONS[q_idx]
            pair_a = normalize_pair_name(pair_a)
            pair_b = normalize_pair_name(pair_b)

            # Add pairs to index if not already there
            for pair in [pair_a, pair_b]:
                if pair not in pair_to_idx:
                    idx = len(pair_to_idx)
                    pair_to_idx[pair] = idx
                    idx_to_pair[idx] = pair

            # Get selected pair from response
            col_idx = q_idx + 1  # +1 to skip timestamp column
            if col_idx < len(row):
                selected_pair = normalize_pair_name(str(row.iloc[col_idx]))

                # Record the comparison
                if selected_pair == pair_a:
                    comparisons[pair_a][pair_b] += 1
                elif selected_pair == pair_b:
                    comparisons[pair_b][pair_a] += 1

    return comparisons, pair_to_idx, idx_to_pair

def bradley_terry_mle(comparisons, pair_to_idx, idx_to_pair):
    """
    Compute Bradley-Terry model rankings using Maximum Likelihood Estimation.

    Returns:
        Dictionary mapping pair names to their strength parameters
    """
    n = len(pair_to_idx)

    # Build win matrix
    wins = np.zeros((n, n))
    for pair_i, opponents in comparisons.items():
        i = pair_to_idx[pair_i]
        for pair_j, count in opponents.items():
            j = pair_to_idx[pair_j]
            wins[i, j] = count

    # Negative log-likelihood for Bradley-Terry model
    def neg_log_likelihood(log_strengths):
        strengths = np.exp(log_strengths)
        nll = 0
        for i in range(n):
            for j in range(n):
                if wins[i, j] > 0:
                    p_i_beats_j = strengths[i] / (strengths[i] + strengths[j])
                    nll -= wins[i, j] * np.log(p_i_beats_j + 1e-10)
        return nll

    # Initial guess: log(1) = 0 for all items
    x0 = np.zeros(n)

    # Optimize
    result = minimize(neg_log_likelihood, x0, method='BFGS')

    # Convert back to strengths
    log_strengths = result.x
    strengths = np.exp(log_strengths)

    # Normalize so they sum to n (for interpretability)
    strengths = strengths * n / strengths.sum()

    # Create dictionary
    strength_dict = {idx_to_pair[i]: strengths[i] for i in range(n)}

    return strength_dict

def plot_pair_rankings(strengths, output_path='object_pair_rankings.png'):
    """Create a bar chart showing risk ranking of object pairs using Bradley-Terry scores"""

    # Sort pairs by Bradley-Terry strength (ascending order)
    # Least risky (lowest strength) at top, most risky at bottom
    sorted_pairs = sorted(strengths.items(), key=lambda x: x[1], reverse=False)

    pairs = [p[0] for p in sorted_pairs]
    scores = [p[1] for p in sorted_pairs]

    # Create figure with more left margin and height for labels
    fig, ax = plt.subplots(figsize=(14, max(10, len(pairs) * 0.8)))
    fig.subplots_adjust(left=0.35)  # Increase left margin significantly for y-axis labels

    # Color gradient from green (least risky) at bottom to red (most risky) at top
    # Reverse because we're plotting in ascending order
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(pairs)))

    # Create horizontal bar chart with more spacing
    y_pos = np.arange(len(pairs))
    ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.7)

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs, fontsize=9, rotation=0, ha='right', va='center')
    ax.set_ylim(-0.5, len(pairs) - 0.5)  # Add padding at top and bottom
    ax.set_xlabel('Bradley-Terry Risk Score', fontsize=12, fontweight='bold')
    ax.set_title('Object Pair Risk Rankings\n(Bradley-Terry Model from Pairwise Comparisons)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (pair, score) in enumerate(sorted_pairs):
        ax.text(score + 0.05, i, f'{score:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on right for value labels
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

    return fig

def print_summary(strengths, comparisons):
    """Print summary of rankings"""
    print("\n" + "="*80)
    print("OBJECT PAIR RISK RANKINGS (Bradley-Terry Model)")
    print("="*80)
    print("\nRanked by Bradley-Terry strength parameter (higher = riskier):\n")

    sorted_pairs = sorted(strengths.items(), key=lambda x: x[1], reverse=True)

    # Count total wins for reference
    total_wins = {}
    for pair in strengths.keys():
        total_wins[pair] = sum(comparisons[pair].values())

    for rank, (pair, strength) in enumerate(sorted_pairs, 1):
        wins = total_wins.get(pair, 0)
        print(f"{rank:2d}. {pair:40s} - Score: {strength:6.2f}  (Wins: {wins})")

    print("\n" + "="*80)

if __name__ == '__main__':
    csv_path = '/home/admin/Projects/bayesian_risk-website/scripts/object_pairs_survey.csv'

    print("Analyzing object pair pairwise comparisons using Bradley-Terry model...")
    comparisons, pair_to_idx, idx_to_pair = analyze_pairwise_comparisons(csv_path)

    print(f"\nFound {len(pair_to_idx)} unique object pairs")
    print("Computing Bradley-Terry rankings...")

    # Compute Bradley-Terry strengths
    strengths = bradley_terry_mle(comparisons, pair_to_idx, idx_to_pair)

    # Print summary
    print_summary(strengths, comparisons)

    # Generate visualization
    print("\nGenerating visualization...")
    plot_pair_rankings(strengths)

    print("\nAnalysis complete!")
