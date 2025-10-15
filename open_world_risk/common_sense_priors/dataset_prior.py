#!/usr/bin/env python3
"""
PriorDataset for handling object pair safety priors.

This dataset is designed for training models on object pair safety ratings
where the target is a single prior value (0-1) instead of histogram distributions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
from datetime import datetime
import numpy as np
from collections import Counter
import random

class PriorDataset(Dataset):
    """
    Dataset for object pair safety priors.
    
    Expected tensor format: [N, 2049] with layout:
    [featA(1024), featB(1024), prior_value(1)]
    
    Input: concatenated DINO features [featA, featB] -> [2048]
    Target: prior value -> [1] (safety rating 0-1)
    
    This is a simplified version of HistogramDataset that works with single
    prior values instead of histogram distributions.
    
    Args:
        tensor_paths: List of paths to tensor files containing the dataset
        distribution_stats_path: Path to pickle file containing distribution statistics
                                saved from build_dataset.py (required)
        feat_size: Size of individual DINO features (default: 1024)
        random_seed: Random seed for reproducible sampling (default: 42)
    """
    
    def __init__(self, tensor_paths: list, distribution_stats_path: str, feat_size: int = 1024, random_seed: int = 42):
        super().__init__()
        
        # Load distribution stats (required)
        if not os.path.exists(distribution_stats_path):
            raise FileNotFoundError(f"Distribution stats file not found: {distribution_stats_path}")
        
        import pickle
        with open(distribution_stats_path, 'rb') as f:
            self.distribution_stats = pickle.load(f)
        print(f"Loaded distribution stats from: {distribution_stats_path}")
        
        # Validate tensor paths and get file info without loading data
        self.tensor_paths = []
        self.file_sizes = []  # Number of samples in each file
        total_samples = 0
        
        for path in tensor_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tensor file not found: {path}")
            
            # Load just to check format and get size, then unload
            tensor = torch.load(path, map_location="cpu")
            if not (isinstance(tensor, torch.Tensor) and tensor.ndim == 2):
                raise ValueError(f"Expected 2D tensor in {path}, got {tensor.shape}")
            
            # Expected format: [N, 2049] = [featA(1024) + featB(1024) + prior(1)]
            expected_cols = (feat_size * 2) + 1
            if tensor.shape[1] != expected_cols:
                raise ValueError(f"Expected {expected_cols} columns in {path}, got {tensor.shape[1]}")
            
            self.tensor_paths.append(path)
            self.file_sizes.append(tensor.shape[0])
            total_samples += tensor.shape[0]
            #print(f"Found {tensor.shape[0]} samples in {path}")
            
            # Clear tensor from memory
            del tensor
        
        print(f"Total dataset size: {total_samples} samples across {len(self.tensor_paths)} files")
        
        self.feat_size = feat_size
        self.total_len = (feat_size * 2) + 1
        self.total_samples = total_samples
        
        # Define slice indices
        self.featA_start, self.featA_end = 0, feat_size
        self.featB_start, self.featB_end = feat_size, feat_size * 2
        self.prior_start, self.prior_end = feat_size * 2, self.total_len
        
        # Cache for loaded tensors (file_path -> tensor)
        self.tensor_cache = {}
        self.max_cache_size = 5  # Keep max 5 files in memory at once
        self.rng = random.Random(random_seed)
    
    def __len__(self):
        return self.total_samples
    
    def _load_tensor(self, file_path):
        """Load tensor from file with caching."""
        if file_path in self.tensor_cache:
            return self.tensor_cache[file_path]
        
        # Load tensor
        tensor = torch.load(file_path, map_location="cpu")
        
        # Manage cache size
        if len(self.tensor_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.tensor_cache))
            del self.tensor_cache[oldest_key]
        
        self.tensor_cache[file_path] = tensor
        return tensor
    
    def __getitem__(self, idx):
        # For balanced sampling, we need deterministic access by index
        # For regular sampling, we can use random access
        
        # Find which file contains this index
        cumulative_samples = 0
        file_idx = 0
        for i, file_size in enumerate(self.file_sizes):
            if idx < cumulative_samples + file_size:
                file_idx = i
                break
            cumulative_samples += file_size
        
        # Calculate the sample index within the file
        sample_idx = idx - cumulative_samples
        
        # Load the tensor for this file
        file_path = self.tensor_paths[file_idx]
        tensor = self._load_tensor(file_path)
        
        # Get the specific sample
        row = tensor[sample_idx]
        
        # Extract features and prior value
        featA = row[self.featA_start:self.featA_end].float()
        featB = row[self.featB_start:self.featB_end].float()
        prior_value = row[self.prior_start:self.prior_end].float()
        
        # Always predict prior from features
        inputs = torch.cat([featA, featB], dim=0)  # [2048]
        target = prior_value  # [1]
        return inputs, target
    
    def get_feature_dims(self):
        """Get input and target dimensions for this dataset"""
        return 2*self.feat_size, 1  # input_dim, target_dim
    
    def get_distribution_stats(self):
        """Get the loaded distribution statistics."""
        return self.distribution_stats
    
    def get_prior_counts(self):
        """Get the prior value counts from distribution stats."""
        if 'prior_counts' not in self.distribution_stats:
            raise ValueError("Distribution stats missing 'prior_counts' key")
        return self.distribution_stats['prior_counts']
    
    def get_rating_counts(self):
        """Get the safety rating counts from distribution stats."""
        if 'rating_counts' not in self.distribution_stats:
            raise ValueError("Distribution stats missing 'rating_counts' key")
        return self.distribution_stats['rating_counts']
    
    def get_distribution_summary(self):
        """Get a summary of the dataset distribution statistics."""
        summary = {
            'total_pairs': self.distribution_stats.get('total_pairs', 'Unknown'),
            'mean_prior': self.distribution_stats.get('mean_prior', 'Unknown'),
            'std_prior': self.distribution_stats.get('std_prior', 'Unknown'),
            'estimated_total_samples': self.distribution_stats.get('estimated_total_samples', 'Unknown'),
            'prior_counts': self.get_prior_counts(),
            'rating_counts': self.get_rating_counts()
        }
        return summary
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: List of (inputs, targets) tuples from __getitem__
            
        Returns:
            inputs: dict with keys:
                - 'dino1': [batch_size, 1024] - first DINO features
                - 'dino2': [batch_size, 1024] - second DINO features
            targets: dict with keys:
                - 'prior': [batch_size, 1] - prior values
        """
        # Extract inputs and targets
        inputs, targets = zip(*batch)
        
        # Split concatenated inputs back into dino1 and dino2
        inputs_tensor = torch.stack(inputs, dim=0)  # [batch_size, 2048]
        dino1 = inputs_tensor[:, :self.feat_size]  # [batch_size, 1024]
        dino2 = inputs_tensor[:, self.feat_size:]  # [batch_size, 1024]
        
        # Stack targets
        priors = torch.stack(targets, dim=0)  # [batch_size, 1]
        
        # Return inputs and targets as dictionaries
        inputs_dict = {
            'dino1': dino1,  # [batch_size, 1024]
            'dino2': dino2   # [batch_size, 1024]
        }
        
        targets_dict = {
            'prior': priors  # [batch_size, 1]
        }
        
        return inputs_dict, targets_dict

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        """
        Create a DataLoader with the custom collate function automatically applied.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)

    def get_balanced_dataloader(self, batch_size=1, subset_indices=None, 
                                target_distribution=None, **kwargs):
        """
        Create a DataLoader with balanced sampling across prior value classes.
        
        Args:
            batch_size: Batch size for the DataLoader
            subset_indices: Optional list/tensor of indices to use for balancing.
                          If None, uses the entire dataset. If provided, only balances
                          the specified subset (useful for train/val splits).
            label_distribution: Dict mapping prior values to their counts in the dataset.
                              e.g., {0.0: 1000, 0.25: 2000, 0.5: 1500, 0.75: 800, 1.0: 1200}
                              If None, uses loaded distribution stats.
            target_distribution: Dict mapping prior values to target counts for balanced sampling.
                               If None, uses equal distribution across all classes.
            **kwargs: Additional arguments passed to DataLoader
        """
        # Use loaded distribution stats (always available)
        print("Using loaded distribution stats for balanced sampling...")
        label_distribution = self.get_prior_counts()
        
        print("Creating balanced DataLoader with weighted sampling...")
        
        # Calculate target distribution if not provided
        if target_distribution is None:
            # Equal distribution across all classes
            unique_labels = list(label_distribution.keys())
            total_samples = sum(label_distribution.values())
            target_count_per_class = total_samples // len(unique_labels)
            target_distribution = {label: target_count_per_class for label in unique_labels}
        
        # Calculate weights for each class
        class_weights = {}
        for label, count in label_distribution.items():
            target_count = target_distribution.get(label, 0)
            if count > 0:
                class_weights[label] = target_count / count
            else:
                class_weights[label] = 0.0
        
        print(f"Estimated label distribution: {label_distribution}")
        print(f"Target distribution: {target_distribution}")
        print(f"Class weights: {class_weights}")
        
        # Create sample weights for WeightedRandomSampler
        # We need to create weights for each sample based on their prior values
        print("Creating sample weights for WeightedRandomSampler...")
        
        # Sample a subset of the dataset to estimate weights for each sample
        sample_size = min(10000, self.total_samples)  # Sample up to 10k samples for efficiency
        sample_indices = self.rng.sample(range(self.total_samples), sample_size)
        
        # Count samples by prior value to create weights
        sample_weights = []
        prior_counts = {}
        
        for idx in sample_indices:
            _, target = self[idx]
            prior_value = target.item()
            
            # Round to nearest 0.25 for grouping (0.0, 0.25, 0.5, 0.75, 1.0)
            rounded_value = round(prior_value * 4) / 4
            prior_counts[rounded_value] = prior_counts.get(rounded_value, 0) + 1
        
        # Create weights for each sample based on class weights
        for idx in sample_indices:
            _, target = self[idx]
            prior_value = target.item()
            rounded_value = round(prior_value * 4) / 4
            
            # Get the weight for this class
            weight = class_weights.get(rounded_value, 1.0)
            sample_weights.append(weight)
        
        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_indices),
            replacement=True
        )
        
        # Create a subset dataset for the sampled indices
        subset_dataset = Subset(self, sample_indices)
        
        # Create DataLoader with WeightedRandomSampler
        return DataLoader(
            subset_dataset,
            batch_size=batch_size, 
            sampler=sampler,
            collate_fn=self.collate_fn, 
            **kwargs
        )

def generate_common_sense_dataset(priors_data: list, output_dir: str = "common_sense_priors", batch_size: int = 50_000):
    """
    Generate dataset from common sense prior data for training.
    
    Args:
        priors_data: List of prior dictionaries from build_priors function.
                     Each dict contains:
                     - 'pair': object pair info with rating
                     - 'prior': single prior value (0-1)
                     - 'dino_a', 'dino_b': DINO embeddings for both objects
        output_dir: Directory to save dataset tensor (default: "common_sense_priors")
        batch_size: Number of samples to process before saving intermediate tensor (default: 50000)
        
    Returns:
        tensor_paths: List of paths to the saved dataset tensor files
    """
    assert len(priors_data) != 0, "Empty priors data passed in"
    print(f"Processing {len(priors_data)} common sense priors in batches of {batch_size}...")
    total_combinations = len(priors_data)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    tensor_paths = []
    
    # Process data in batches
    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_data = priors_data[batch_start:batch_end]
        batch_size_actual = len(batch_data)
        
        print(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start} to {batch_end-1} ({batch_size_actual} samples)")
        
        # Pre-allocate tensor for this batch: [N, 2049] = [featA(1024) + featB(1024) + prior(1)]
        batch_dataset = torch.empty((batch_size_actual, 2049), dtype=torch.float32)
        
        for idx, prior_dict in enumerate(tqdm(batch_data, desc=f"Processing batch {batch_start//batch_size + 1}")):
            dino_a = prior_dict['dino_a']  # Should be torch.Tensor
            dino_b = prior_dict['dino_b']  # Should be torch.Tensor
            prior_value = prior_dict['prior']  # Single float value
            
            if not isinstance(dino_a, torch.Tensor):
                dino_a = torch.tensor(dino_a, dtype=torch.float32)
            if not isinstance(dino_b, torch.Tensor):
                dino_b = torch.tensor(dino_b, dtype=torch.float32)
            if not isinstance(prior_value, torch.Tensor):
                prior_value = torch.tensor(prior_value, dtype=torch.float32)

            assert dino_a.shape[0] == 1024, f"DINO embedding A has wrong dimension: {dino_a.shape[0]}, expected 1024"
            assert dino_b.shape[0] == 1024, f"DINO embedding B has wrong dimension: {dino_b.shape[0]}, expected 1024"
            assert 0.0 <= prior_value.item() <= 1.0, f"Prior value {prior_value.item()} is outside [0, 1] range"
            
            batch_dataset[idx, :1024] = dino_a.cpu()
            batch_dataset[idx, 1024:2048] = dino_b.cpu()
            batch_dataset[idx, 2048] = prior_value 
        
        # Save this batch
        batch_tensor_path = os.path.join(output_dir, f"common_sense_prior_tensor_{timestamp}_batch_{batch_start//batch_size + 1}.pt")
        torch.save(batch_dataset, batch_tensor_path)
        tensor_paths.append(batch_tensor_path)
        print(f"Saved batch {batch_start//batch_size + 1} with {batch_dataset.shape[0]} samples to {batch_tensor_path}")
    
    print(f"Created {len(tensor_paths)} batch files with total {total_combinations} samples")
    print(f"All batch files saved to {output_dir}")
    
    return tensor_paths

def test_prior_dataset(tensor_paths: list[str], distribution_stats_path, batch_size: int = 4, output_dir: str = "."):
    """
    Test function for PriorDataset to verify functionality and data integrity.
    
    Args:
        tensor_paths: List of paths to the .pt files containing the dataset tensors
        distribution_stats: Dictionary containing distribution statistics from create_summary_plot
        batch_size: Batch size for testing DataLoader
        output_dir: Directory to save test plots (default: current directory)
    
    Returns:
        dict: Test results including dataset info, sample data, and validation results
    """
    print("=" * 60)
    print("TESTING PriorDataset")
    print("=" * 60)
    
    results = {}
    
    try:
        # Test 1: Dataset initialization
        print("\n1. Testing dataset initialization...")
        dataset = PriorDataset(tensor_paths, distribution_stats_path)
        results['dataset_loaded'] = True
        results['dataset_size'] = len(dataset)
        results['input_dim'], results['target_dim'] = dataset.get_feature_dims()
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Dataset size: {len(dataset)} samples")
        print(f"  - Input dimension: {results['input_dim']}")
        print(f"  - Target dimension: {results['target_dim']}")
        
        # Test 2: Single sample access
        print("\n2. Testing single sample access...")
        if len(dataset) > 0:
            sample_input, sample_target = dataset[0]
            results['sample_input_shape'] = sample_input.shape
            results['sample_target_shape'] = sample_target.shape
            
            print(f"✓ Sample access successful")
            print(f"  - Input shape: {sample_input.shape}")
            print(f"  - Target shape: {sample_target.shape}")
            print(f"  - Input range: [{sample_input.min():.4f}, {sample_input.max():.4f}]")
            print(f"  - Target range: [{sample_target.min():.4f}, {sample_target.max():.4f}]")
            
            # Validate tensor format
            if sample_input.shape[0] != 2048:
                raise ValueError(f"Expected input size 2048, got {sample_input.shape[0]}")
            if sample_target.shape[0] != 1:
                raise ValueError(f"Expected target size 1, got {sample_target.shape[0]}")
            
            results['tensor_format_valid'] = True
            print("✓ Tensor format validation passed")
        else:
            print("⚠ Dataset is empty")
            results['sample_input_shape'] = None
            results['sample_target_shape'] = None
            results['tensor_format_valid'] = False
        
        # Test 3: Label Distribution Analysis
        print("\n3. Analyzing label distribution...")
        if len(dataset) > 0:
            # Use provided distribution from build_dataset.py
            print("Using provided distribution from original data...")
            provided_distribution = dataset.distribution_stats['prior_counts']
            
            print(f"Label distribution from original safety ratings:")
            for prior_val in sorted(provided_distribution.keys()):
                count = provided_distribution[prior_val]
                percentage = (count / sum(provided_distribution.values())) * 100
                print(f"  Prior {prior_val:.2f}: {count} samples ({percentage:.1f}%)")
            
            results['label_distribution'] = provided_distribution
            results['total_pairs'] = dataset.distribution_stats['total_pairs']
            results['mean_prior'] = dataset.distribution_stats['mean_prior']
            results['std_prior'] = dataset.distribution_stats['std_prior']
            print("✓ Label distribution analysis completed")
        
        # Test 4: Balanced Sampling Distribution Test
        print("\n4. Testing balanced sampling distribution...")
        if len(dataset) > 0:
            # Create balanced dataloader
            balanced_loader = dataset.get_balanced_dataloader(batch_size=batch_size)
            
            # Sample from balanced dataloader to check distribution
            sample_size = min(1000, len(dataset))  # Sample up to 1000 samples
            sampled_priors = []
            
            print(f"Sampling {sample_size} samples from balanced dataloader...")
            sample_count = 0
            for batch_inputs, batch_targets in balanced_loader:
                priors = batch_targets['prior'].squeeze()
                sampled_priors.extend(priors.tolist())
                sample_count += len(priors)
                
                if sample_count >= sample_size:
                    break
            
            # Truncate to exact sample size
            sampled_priors = sampled_priors[:sample_size]
            
            # Count distribution of sampled priors
            sampled_distribution = {}
            for prior_val in sampled_priors:
                # Round to nearest 0.25 for grouping
                rounded_val = round(prior_val * 4) / 4
                sampled_distribution[rounded_val] = sampled_distribution.get(rounded_val, 0) + 1
            
            print(f"Sampled distribution from balanced dataloader:")
            for prior_val in sorted(sampled_distribution.keys()):
                count = sampled_distribution[prior_val]
                percentage = (count / len(sampled_priors)) * 100
                print(f"  Prior {prior_val:.2f}: {count} samples ({percentage:.1f}%)")
            
            # Compare with original distribution
            print(f"\nComparison with original distribution:")
            original_total = sum(provided_distribution.values())
            for prior_val in sorted(set(provided_distribution.keys()) | set(sampled_distribution.keys())):
                original_count = provided_distribution.get(prior_val, 0)
                sampled_count = sampled_distribution.get(prior_val, 0)
                original_pct = (original_count / original_total) * 100
                sampled_pct = (sampled_count / len(sampled_priors)) * 100
                print(f"  Prior {prior_val:.2f}: Original {original_pct:.1f}% → Sampled {sampled_pct:.1f}%")
            
            results['sampled_distribution'] = sampled_distribution
            results['sampled_size'] = len(sampled_priors)
            print("✓ Balanced sampling distribution test completed")
        
        # Test 5: DataLoader with custom collate function
        print("\n5. Testing DataLoader with custom collate function...")
        if len(dataset) > 0:
            # Create a small batch for testing
            test_batch_size = min(batch_size, len(dataset))
            dataloader = dataset.get_dataloader(
                batch_size=test_batch_size, 
                shuffle=False
            )
            
            # Get one batch
            batch_inputs, batch_targets = next(iter(dataloader))
            
            results['batch_loaded'] = True
            results['batch_size'] = test_batch_size
            results['dino1_shape'] = batch_inputs['dino1'].shape
            results['dino2_shape'] = batch_inputs['dino2'].shape
            results['prior_shape'] = batch_targets['prior'].shape
            
            print(f"✓ DataLoader test successful")
            print(f"  - Batch size: {test_batch_size}")
            print(f"  - DINO1 shape: {batch_inputs['dino1'].shape}")
            print(f"  - DINO2 shape: {batch_inputs['dino2'].shape}")
            print(f"  - Prior shape: {batch_targets['prior'].shape}")
            
            # Validate shapes
            expected_dino_shape = (test_batch_size, 1024)
            expected_prior_shape = (test_batch_size, 1)
            
            if batch_inputs['dino1'].shape != expected_dino_shape:
                raise ValueError(f"Expected dino1 shape {expected_dino_shape}, got {batch_inputs['dino1'].shape}")
            if batch_inputs['dino2'].shape != expected_dino_shape:
                raise ValueError(f"Expected dino2 shape {expected_dino_shape}, got {batch_inputs['dino2'].shape}")
            if batch_targets['prior'].shape != expected_prior_shape:
                raise ValueError(f"Expected prior shape {expected_prior_shape}, got {batch_targets['prior'].shape}")
            
            results['batch_shapes_valid'] = True
            print("✓ Batch shape validation passed")
            
            # Test data consistency
            print("\n5. Testing data consistency...")
            # With random sampling, we can't test exact consistency, but we can test
            # that the reconstructed input has the correct shape and format
            reconstructed_input = torch.cat([batch_inputs['dino1'], batch_inputs['dino2']], dim=1)
            
            # Check that reconstructed input has correct shape [batch_size, 2048]
            if reconstructed_input.shape == (test_batch_size, 2048):
                results['data_consistency'] = True
                print("✓ Data consistency check passed (shape validation)")
            else:
                results['data_consistency'] = False
                print(f"⚠ Data consistency check failed: expected shape ({test_batch_size}, 2048), got {reconstructed_input.shape}")
            
            # Test prior value validity
            print("\n6. Testing prior value validity...")
            priors = batch_targets['prior']
            valid_priors = True
            
            for i in range(test_batch_size):
                prior = priors[i].item()
                
                # Check for NaN or Inf values
                if torch.isnan(priors[i]) or torch.isinf(priors[i]):
                    print(f"⚠ Prior value for sample {i} is NaN or Inf")
                    valid_priors = False
                    continue
                
                # Check if prior is in valid range [0, 1]
                if not (0.0 <= prior <= 1.0):
                    print(f"⚠ Prior value for sample {i} out of range [0,1]: {prior:.6f}")
                    valid_priors = False
                else:
                    print(f"✓ Prior value for sample {i} in valid range: {prior:.6f}")
            
            results['prior_valid'] = valid_priors
            if valid_priors:
                print("✓ Prior value validation passed")
            else:
                print("⚠ Prior value validation failed")
                
        else:
            print("⚠ Cannot test DataLoader - dataset is empty")
            results['batch_loaded'] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        all_tests_passed = all([
            results.get('dataset_loaded', False),
            results.get('tensor_format_valid', False),
            results.get('batch_loaded', False),
            results.get('batch_shapes_valid', False),
            results.get('data_consistency', False),
            results.get('prior_valid', False)
        ])
        
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
            failed_tests = [k for k, v in results.items() if isinstance(v, bool) and not v]
            print(f"Failed tests: {failed_tests}")
        
        results['all_tests_passed'] = all_tests_passed
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception: {e}")
        results['error'] = str(e)
        results['all_tests_passed'] = False
    
    return results
