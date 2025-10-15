#!/usr/bin/env python3
"""
Build dataset script to extract DINO features for objects and create averaged feature dictionary.
This script processes images from prior_GPT_data_09_14_2025 and creates a dictionary mapping
object labels to their averaged DINO features.
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
import sys
import yaml
from datetime import datetime
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import Counter
import pickle

from open_world_risk.src.dino_features import DinoFeatures
from open_world_risk.common_sense_priors.dataset_prior import (
    generate_common_sense_dataset, 
    test_prior_dataset
)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_output_directory(base_output_dir="output"):
    """Create output directory with naming convention: prior_{date_month_day_year}_timestamp"""
    now = datetime.now()
    date_str = now.strftime("%m_%d_%Y")
    timestamp = int(now.timestamp())
    output_dir_name = f"prior_{date_str}_{timestamp}"
    output_dir_path = os.path.join(base_output_dir, output_dir_name)
    
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path

def load_object_labels(csv_path):
    """Load object labels from the CSV file."""
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                labels.append(row[0].strip())
    return labels

def get_image_files(object_dir, image_extensions=None):
    """Get all image files from an object directory."""
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    image_files = []
    
    if not os.path.exists(object_dir):
        raise FileNotFoundError(f"Object directory does not exist: {object_dir}")

    for file in os.listdir(object_dir):
        if any(file.endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(object_dir, file))
    
    return image_files

def labels_in_obj_ims(image_dir, labels):
    """Check which labels from CSV are missing from object_images directory."""
    missing_labels = []
    present_labels = []
    
    for label in labels:
        # Convert label to directory name (replace spaces with underscores)
        object_dir_name = label.replace(" ", "_")
        object_dir = os.path.join(image_dir, object_dir_name)
        
        if os.path.exists(object_dir):
            # Check if directory has any files
            files = os.listdir(object_dir)
            if files:
                present_labels.append(label)
            else:
                missing_labels.append(label)
        else:
            missing_labels.append(label)

    print(f"Labels with images: {len(present_labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    if missing_labels:
        print("Missing labels:")
        for i in range(0, len(missing_labels), 5):
            batch = missing_labels[i:i+5]
            print("  " + ", ".join(batch))
    
    return present_labels, missing_labels

def load_images_np(image_files):
    assert image_files, f"No image files found"
    
    images_np = []
    for image_path in image_files:
        image = Image.open(image_path).convert('RGB')
        images_np.append(np.array(image))
    
    return images_np

def extract_dino_features_for_object(dino_model, images_np, target_h=224, use_global_average_pooling=True, num_rand=10):
    """Extract DINO features for all images of a single object and return averaged features."""
    assert images_np, f"No images provided"
    
    features_list = []
    for image_np in images_np:
        rgb_tensor = dino_model.rgb_image_preprocessing(image_np, target_h)
        dino_features = dino_model.get_dino_features_rgb(rgb_tensor)
        
        # Global average pooling to get a single feature vector per image
        # dino_features shape: [C, H, W], we want [C]
        if use_global_average_pooling:
            global_feature = torch.mean(dino_features, dim=(1, 2))  # Average over spatial dimensions
            features_list.append(global_feature.cpu())
            assert features_list, f"No valid features extracted for this object"
    
            # Average features across all images for this object
            features_list= [torch.stack(features_list).mean(dim=0)]
            
        else:
            # Randomly sample num_rand features from spatial locations
            # dino_features shape: [C, H, W]
            C, H, W = dino_features.shape
            total_locations = H * W
            
            # Flatten spatial dimensions to sample from
            flattened_features = dino_features.view(C, -1)  # [C, H*W]
            
            # Randomly sample locations
            sample_size = min(num_rand, total_locations)
            sampled_indices = torch.randperm(total_locations)[:sample_size]
            
            # Extract features at sampled locations and add to list
            for idx in sampled_indices:
                sampled_feature = flattened_features[:, idx]  # [C]
                features_list.append(sampled_feature.cpu())
            assert features_list, f"No valid features extracted for this object"

        return features_list


def build_object_feature_dictionary(config, output_dir, use_global_average_pooling=True, num_rand=10):
    """Build the complete object feature dictionary."""
    
    dino_model = DinoFeatures()
    data_dir = config['data']['data_dir']
    objects_csv_path = os.path.join(data_dir, config['data']['objects_csv'])
    object_images_dir = os.path.join(data_dir, config['data']['object_images_dir'])
    
    print("Loading object labels...")
    labels = load_object_labels(objects_csv_path)
    print(f"Found {len(labels)} object labels")
    
    # Check which labels have images
    present_labels, missing_labels = labels_in_obj_ims(object_images_dir, labels)
    
    # Build feature dictionary
    feature_dict = {}
    processed_count = 0
    
    print("Loading all images...")
    all_images_data = []
    for i, label in enumerate(tqdm(present_labels, desc="Loading images")):
        object_dir_name = label.replace(" ", "_")
        object_dir = os.path.join(object_images_dir, object_dir_name)
        image_files = get_image_files(object_dir, config['processing']['image_extensions'])
        assert len(image_files) > 0, "No image files found"
        
        images_np = load_images_np(image_files)
        all_images_data.append((label, images_np))
    
    print("Processing all images...")
    for i, (label, images_np) in enumerate(tqdm(all_images_data, desc="Processing objects")):
        averaged_features = extract_dino_features_for_object(dino_model, images_np, config['dino']['target_height'],
                                                            use_global_average_pooling, num_rand)       
        feature_dict[label] = averaged_features
        processed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {processed_count}/{len(labels)} objects")
    
    # Also save as a summary
    summary_path = os.path.join(output_dir, config['data']['output_summary'])
    with open(summary_path, 'w') as f:
        f.write(f"DINO Feature Extraction Summary\n")
        f.write(f"==============================\n")
        f.write(f"Total objects: {len(labels)}\n")
        f.write(f"Successfully processed: {processed_count}\n")
        f.write(f"Failed: {len(labels) - processed_count}\n")
        f.write(f"\nProcessed objects:\n")
        for label in feature_dict.keys():
            f.write(f"  - {label}\n")
        f.write(f"\nFailed objects:\n")
        for label in labels:
            if label not in feature_dict:
                f.write(f"  - {label}\n")
    
    print(f"Summary saved to {summary_path}")
    
    return feature_dict

def create_summary_plot(valid_pairs, output_dir):
    """Create a summary plot showing distribution of safety ratings and prior values."""
    print("Creating summary plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count ratings
    rating_counts = {}
    for pair in valid_pairs:
        rating = pair['rating']
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    # Create bar plot for ratings
    ratings = sorted(rating_counts.keys())
    counts = [rating_counts[r] for r in ratings]
    
    bars1 = ax1.bar(ratings, counts, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    ax1.set_xlabel('Safety Rating (1=unsafe, 5=safe)')
    ax1.set_ylabel('Number of Object Pairs')
    ax1.set_title('Distribution of Safety Ratings for Object Pairs')
    ax1.set_xticks(ratings)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Calculate and count prior values
    prior_values = []
    for pair in valid_pairs:
        prior_value = (pair['rating'] - 1) / 4.0
        prior_values.append(prior_value)
    
    # Create histogram for prior values
    ax2.hist(prior_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Prior Value')
    ax2.set_ylabel('Number of Object Pairs')
    ax2.set_title('Distribution of Prior Values')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    mean_prior = np.mean(prior_values)
    std_prior = np.std(prior_values)
    ax2.text(0.02, 0.98, f'Mean: {mean_prior:.3f}\nStd: {std_prior:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, "safety_ratings_summary.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to {summary_plot_path}")
    
    # Calculate and return distribution statistics for sampling
    prior_counts = Counter([round(p, 2) for p in prior_values])
    total_pairs = len(valid_pairs)
    
    # Scale up to estimate full dataset distribution (accounting for feature combinations)
    # Each pair creates multiple combinations based on feature lists
    estimated_total_samples = total_pairs  # This will be updated when we know the actual combinations
    
    distribution_stats = {
        'rating_counts': rating_counts,
        'prior_counts': dict(prior_counts),
        'total_pairs': total_pairs,
        'mean_prior': mean_prior,
        'std_prior': std_prior,
        'estimated_total_samples': estimated_total_samples
    }
    
    print(f"Distribution statistics:")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Mean prior: {mean_prior:.3f}")
    print(f"  Std prior: {std_prior:.3f}")
    print(f"  Prior distribution: {dict(prior_counts)}")
    
    return distribution_stats

def test_all_priors(all_priors):
    """
    Test function to print sample data from all_priors.
    Shows first 10 data points with truncated DINO embeddings.
    """
    print("\n" + "="*80)
    print("TESTING ALL_PRIORS - Sample Data Points")
    print("="*80)
    
    # Show first 10 data points
    for i, prior_data in enumerate(all_priors[:10]):
        print(f"\n--- Data Point {i+1} ---")
        print(f"Object A: {prior_data['pair']['object_a']}")
        print(f"Object B: {prior_data['pair']['object_b']}")
        print(f"Rating: {prior_data['pair']['rating']}")
        print(f"Prior Value: {prior_data['prior']:.3f}")
        print(f"DINO A shape: {prior_data['dino_a'].shape}")
        print(f"DINO A (first 4): {prior_data['dino_a'][:4]}")
        print(f"DINO B shape: {prior_data['dino_b'].shape}")
        print(f"DINO B (first 4): {prior_data['dino_b'][:4]}")
    
    print(f"\nTotal priors created: {len(all_priors)}")
    print("="*80)

def build_priors(config, feature_dict, output_dir, sample_size):    
    data_dir = config['data']['data_dir']
    object_pairs_csv_path = os.path.join(data_dir, config['data']['object_pairs_csv'])
    
    print("Loading object pair safety ratings...")
    
    # Read the CSV file
    valid_pairs = []
    with open(object_pairs_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_a = row['object_a'].strip()
            object_b = row['object_b'].strip()
            rating = int(row['rating_1_unsafe_5_safe'])
            
            # Check if both objects are in feature_dict
            if object_a in feature_dict and object_b in feature_dict:
                valid_pairs.append({
                    'object_a': object_a,
                    'object_b': object_b,
                    'rating': rating,
                    'reason': row['reason']
                })
    
    print(f"Found {len(valid_pairs)} valid object pairs (both objects have features)")
    
    # Create priors for all valid pairs
    print("Creating priors for all valid pairs...")
    all_priors = []
    
    for pair in tqdm(valid_pairs, desc="Creating priors"):
        # Calculate y value based on rating
        prior_value = (pair['rating'] - 1) / 4.0
        dino_a_list = feature_dict[pair['object_a']]  # Now a list of features
        dino_b_list = feature_dict[pair['object_b']]  # Now a list of features

        # Create all combinations of features from the two lists
        # Randomly sample num_rand combinations of features
        num_rand = config['dino']['num_rand']
        num_combinations = min(num_rand, len(dino_a_list) * len(dino_b_list))
        
        # Create all possible combinations and randomly sample from them
        all_combinations = [(dino_a, dino_b) for dino_a in dino_a_list for dino_b in dino_b_list]
        sampled_combinations = random.sample(all_combinations, num_combinations)
        
        for dino_a, dino_b in sampled_combinations:
            all_priors.append({
                'pair': pair,
                'prior': prior_value, 
                'dino_a': dino_a,
                'dino_b': dino_b
            })
    
    print(f"Created {len(all_priors)} priors")
    
    # Randomly sample priors for plotting
    actual_sample_size = min(sample_size, len(all_priors))
    sampled_priors = random.sample(all_priors, actual_sample_size)
    print(f"Randomly sampled {actual_sample_size} priors for plotting")
    
    # Create summary plot and get distribution stats
    distribution_stats = create_summary_plot(valid_pairs, output_dir)
    
    # Update estimated total samples with actual count
    distribution_stats['estimated_total_samples'] = len(all_priors)
    
    return all_priors, sampled_priors, distribution_stats

def main():
    """Main function to run the feature extraction."""
    config_path = os.path.join(os.path.dirname(__file__), "build_dataset_config.yaml")
    config = load_config(config_path)
    
    output_dir = create_output_directory()
    print(f"Created output directory: {output_dir}")
    
    config_dest_path = os.path.join(output_dir, "build_dataset_config.yaml")
    shutil.copy2(config_path, config_dest_path)
    print(f"Copied config to output directory: {config_dest_path}")
    
    print(f"Building object feature dictionary using config from {config_path}")
    feature_dict = build_object_feature_dictionary(config, output_dir, config["dino"]["use_global_average_pooling"], 
                                                    config["dino"]['num_rand'])
    sample_size = config.get('sample_size', 100)
    
    all_priors, sampled_priors, distribution_stats = build_priors(config, feature_dict, output_dir, sample_size)
    test_all_priors(all_priors)

    tensor_paths = generate_common_sense_dataset(all_priors, output_dir, batch_size=200_000)

    # Save distribution stats to pickle file
    distribution_stats_path = os.path.join(output_dir, "distribution_stats.pkl")
    with open(distribution_stats_path, 'wb') as f:
        pickle.dump(distribution_stats, f)
    print(f"Saved distribution stats to: {distribution_stats_path}")
    print(f"Output saved to: {output_dir}")

    test_prior_dataset(tensor_paths, distribution_stats_path, output_dir=output_dir)

if __name__ == "__main__":
    main()