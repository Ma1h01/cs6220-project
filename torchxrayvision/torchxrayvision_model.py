import os,sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns

try:
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])
    d_chex = xrv.datasets.CheX_Dataset(imgpath="/home/hice1/ymai8/scratch/cs6220-project/CheXpert-v1.0-small",
                                       csvpath="/home/hice1/ymai8/scratch/cs6220-project/CheXpert-v1.0-small/valid.csv",
                                     views=["PA","AP"], unique_patients=False, transform=transform)
    
    print("Validation Dataset object (for false positive analysis):")
    print(d_chex)                                 
    print("-" * 30)

    # 1. Load the Model
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    
    # 2. Clean model's labels and dataset labels
    print(f"Model pathologies ({len(model.pathologies)}): {model.pathologies}")
    print(f"Dataset pathologies ({len(d_chex.pathologies)}): {d_chex.pathologies}")
    # Filter out empty pathologies from model and create mapping
    model_pathologies_clean = [p for p in model.pathologies if p.strip()]
    dataset_pathologies_clean = [str(p) for p in d_chex.pathologies]
    print(f"\nCleaned model pathologies ({len(model_pathologies_clean)}): {model_pathologies_clean}")
    print(f"Cleaned dataset pathologies ({len(dataset_pathologies_clean)}): {dataset_pathologies_clean}")
    
    # Create mapping between model indices and dataset indices for matching pathologies
    pathology_mapping = {}  # model_idx -> (dataset_idx, pathology_name)
    for model_idx, model_pathology in enumerate(model.pathologies):
        if model_pathology.strip():  # Skip empty pathologies
            for dataset_idx, dataset_pathology in enumerate(dataset_pathologies_clean):
                if model_pathology.strip().lower() == str(dataset_pathology).strip().lower():
                    pathology_mapping[model_idx] = (dataset_idx, model_pathology)
                    break
    print(f"\nPathology mapping found {len(pathology_mapping)} matches:")
    for model_idx, (dataset_idx, name) in pathology_mapping.items():
        print(f"  Model[{model_idx}] -> Dataset[{dataset_idx}]: {name}")
    
    # Analyze what's missing
    mapped_model_indices = set(pathology_mapping.keys())
    mapped_dataset_indices = set(idx for idx, _ in pathology_mapping.values())
    
    unmapped_model_pathologies = []
    for i, p in enumerate(model.pathologies):
        if p.strip() and i not in mapped_model_indices:
            unmapped_model_pathologies.append((i, p))
    
    unmapped_dataset_pathologies = []
    for i, p in enumerate(d_chex.pathologies):
        if i not in mapped_dataset_indices:
            unmapped_dataset_pathologies.append((i, str(p)))
    
    print(f"\nUnmapped model pathologies ({len(unmapped_model_pathologies)}):")
    for idx, name in unmapped_model_pathologies:
        print(f"  Model[{idx}]: {name}")
    
    print(f"\nUnmapped dataset pathologies ({len(unmapped_dataset_pathologies)}):")
    for idx, name in unmapped_dataset_pathologies:
        print(f"  Dataset[{idx}]: {name}")
    
    # Strategy selection based on mapping success
    if len(pathology_mapping) == 0:
        print("\n WARNING: No matching pathologies found! End.")
        sys.exit(0)
    else:
        print(f"\n✓ Good mapping success: {len(pathology_mapping)} pathologies matched")

    # 3. Set up device (use GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval() # Set model to evaluation mode (important for inference)
    print(f"Model moved to: {device}")

    # 4. Create a DataLoader to batch the data
    dataloader_config = {
        'batch_size': 4,     # Smaller batch size to avoid memory issues
        'shuffle': False,    # No need to shuffle for inference
        'num_workers': 2    
        }
    dataloader = DataLoader(d_chex, **dataloader_config)   
    
    print("Created DataLoader with batch size {} and num_workers {}".format(dataloader_config['batch_size'], dataloader_config['num_workers']))
    print("-" * 30)

    # 5. Identify False Positives on Validation Set
    print("Analyzing predictions on validation set to identify false positives...")
    
    false_positives = []  # Store false positive cases
    no_finding_count = 0  # Count of "No Finding" cases skipped
    threshold = 0.625  # You can adjust this threshold based on your needs
    
    # Store cached predictions and ground truth for comprehensive metrics (avoids second inference)
    cached_predictions = []  # Store predictions per batch
    cached_ground_truth = []  # Store ground truth per batch
    cached_paths = []  # Store image paths per batch
    
    # Use torch.no_grad() to disable gradient calculations
    with torch.no_grad():
        # Process ALL batches for comprehensive analysis
        total_batches = len(dataloader)
        print(f"Processing all {total_batches} batches...")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:  # Progress indicator every 10 batches
                print(f"  Progress: {batch_idx+1}/{total_batches} batches")
                
            # Get images and ground truth labels
            images = batch['img'].to(device)
            ground_truth = batch['lab'].to(device)  # Ground truth labels
            
            # Run predictions (single inference pass)
            outputs = model(images)
            predictions = torch.sigmoid(outputs)  # Convert to probabilities with sigmoid since multiple pathologies can coexist
            # Convert to binary predictions using threshold
            binary_predictions = (predictions > threshold).float()
            
            # Cache predictions and ground truth for comprehensive metrics later
            cached_predictions.append(predictions.detach().cpu())
            cached_ground_truth.append(ground_truth.detach().cpu())

            # Extract image paths properly from batch
            batch_paths = []
            for i in range(images.shape[0]):
                actual_idx = batch_idx * dataloader_config['batch_size'] + i
                if actual_idx < len(d_chex.csv):
                    batch_paths.append(d_chex.csv.iloc[actual_idx]['Path'])
                else:
                    batch_paths.append("Unknown")
            cached_paths.extend(batch_paths)
            
            # Find false positives: model predicts positive (1) but ground truth is negative (0)
            # Note: In CheXpert, labels can be 0 (negative), 1 (positive), or NaN (uncertain)
            # We'll focus on clear cases where ground truth is 0 and prediction is 1
            
            print(f"Batch {batch_idx}: Images shape: {images.shape}, GT shape: {ground_truth.shape}, Pred shape: {predictions.shape}")
            
            # Since GT and model output could have different pathologies, we need to map them correctly
            for i in range(images.shape[0]):  # For each image in batch
                # Get image info
                actual_image_idx = batch_idx * dataloader_config['batch_size'] + i
                
                # Check if this is a "No Finding" case by looking at the original CSV
                is_no_finding = False
                if actual_image_idx < len(d_chex.csv):
                    no_finding_value = d_chex.csv.iloc[actual_image_idx].get('No Finding', 0)
                    is_no_finding = (no_finding_value == 1)
                    image_path = d_chex.csv.iloc[actual_image_idx]['Path']
                else:
                    image_path = "Unknown"
                
                # Skip "No Finding" cases entirely for false positive analysis
                if is_no_finding:
                    no_finding_count += 1   
                    continue
                
                for model_idx, (dataset_idx, pathology_name) in pathology_mapping.items():
                    if dataset_idx < ground_truth.shape[1] and model_idx < predictions.shape[1]:
                        gt_label = ground_truth[i, dataset_idx].item()
                        pred_prob = predictions[i, model_idx].item()
                        binary_pred = binary_predictions[i, model_idx].item()
                        
                        # Check for false positive: ground truth is 0 (negative) but model predicts positive
                        if gt_label == 0.0 and binary_pred == 1.0:
                            # Use correct image path from batch_paths
                            if i < len(batch_paths):
                                correct_image_path = batch_paths[i]
                            else:
                                correct_image_path = image_path 
                            
                            false_positives.append({
                                'batch_idx': batch_idx,
                                'image_idx': i,
                                'actual_dataset_idx': actual_image_idx,
                                'image_path': correct_image_path,
                                'pathology': pathology_name,
                                'ground_truth': gt_label,
                                'prediction_prob': pred_prob,
                                'binary_prediction': binary_pred,
                                'model_idx': model_idx,
                                'dataset_idx': dataset_idx
                            })
        
        # --- Display Results ---
        print(f"\n=== FALSE POSITIVE ANALYSIS RESULTS ===")
        print(f"Strategy: Analyzing only {len(pathology_mapping)} matched pathologies")
        print(f"Analyzed ALL {total_batches} batches from validation set (complete dataset)")
        print(f"Skipped {no_finding_count} 'No Finding' cases (focusing on pathology misclassification)")
        print(f"Prediction threshold: {threshold}")
        print(f"Total false positives found: {len(false_positives)}")
        print("-" * 60)
        
        if false_positives:
            # Summary by pathology
            # print(f"\n=== FALSE POSITIVES BY PATHOLOGY ===")
            # pathology_counts = {}
            # for fp in false_positives:
            #     pathology = fp['pathology']
            #     pathology_counts[pathology] = pathology_counts.get(pathology, 0) + 1
            
            # for pathology, count in sorted(pathology_counts.items(), key=lambda x: x[1], reverse=True):
            #     print(f"{pathology:<20}: {count} false positives")
            
            print(f"\n=== NOTE ===")
            print(f"Analysis excludes 'No Finding' cases (where No Finding=1 in original data)")

            print(f"\n=== COMPREHENSIVE METRICS (Per Pathology) ===")
            if len(cached_predictions) == 0:
                print("No cached predictions available")
            else:
                # Concatenate all cached predictions and ground truth (no second inference needed!)
                all_preds = torch.cat(cached_predictions, dim=0).numpy()  # Shape: (N_images, num_model_pathologies)
                all_gt = torch.cat(cached_ground_truth, dim=0).numpy()    # Shape: (N_images, num_dataset_pathologies)
                
                print(f"Using cached predictions from single inference pass")
                print(f"Total samples: {all_preds.shape[0]}, Model pathologies: {all_preds.shape[1]}, Dataset pathologies: {all_gt.shape[1]}")
                
                results = []
                false_positive_cases = []
                excluded_pathologies = []  # Track pathologies excluded from metrics

                # Process each pathology using cached data (no model calls needed!)
                for model_idx, (dataset_idx, pathology_name) in pathology_mapping.items():
                    if dataset_idx >= all_gt.shape[1] or model_idx >= all_preds.shape[1]:
                        print(f"[!] Skipping {pathology_name}: index out of bounds")
                        continue
                        
                    # Extract predictions and ground truth for this pathology
                    y_pred_prob = all_preds[:, model_idx]
                    y_true_raw = all_gt[:, dataset_idx]
                    
                    # Filter out NaN ground truth values
                    valid_mask = ~np.isnan(y_true_raw)
                    y_true = y_true_raw[valid_mask]
                    y_pred_prob_valid = y_pred_prob[valid_mask]
                    y_pred_bin = (y_pred_prob_valid > threshold).astype(int)

                    # Skip if insufficient data
                    if len(np.unique(y_true)) < 2:
                        print(f"[!] Skipping {pathology_name}: only one class present in GT")
                        excluded_pathologies.append(pathology_name)
                        continue

                    # Collect false positive cases for this pathology
                    valid_indices = np.where(valid_mask)[0]  # Original indices where mask is True
                    for idx, (gt_val, bin_val, pred_val) in enumerate(zip(y_true, y_pred_bin, y_pred_prob_valid)):
                        if gt_val == 0 and bin_val == 1:
                            original_idx = valid_indices[idx]
                            image_path = cached_paths[original_idx] if original_idx < len(cached_paths) else 'unknown'
                            false_positive_cases.append({
                                'pathology': pathology_name,
                                'prediction_prob': float(pred_val),
                                'ground_truth': int(gt_val),
                                'image_path': image_path,
                                'original_index': int(original_idx)
                            })

                    # Compute metrics
                    try:
                        auc = roc_auc_score(y_true, y_pred_prob_valid)
                    except ValueError:
                        auc = np.nan
                    precision = precision_score(y_true, y_pred_bin, zero_division=0)
                    recall = recall_score(y_true, y_pred_bin, zero_division=0)
                    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

                    # Count false positives for this pathology
                    fp_count = sum(1 for fp_case in false_positive_cases if fp_case['pathology'] == pathology_name)
                    
                    results.append({
                        'Pathology': pathology_name,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'AUROC': auc,
                        'FPR': fpr,
                        'Support': len(y_true),
                        'False_Positives': fp_count
                    })
                
                # Convert to DataFrame for readability
                df_results = pd.DataFrame(results).sort_values(by='AUROC', ascending=False)
                print(df_results.to_string(index=False, float_format="%.4f"))
                
                # Display false positive summary per pathology (sorted by FP count descending)
                print("\n=== FALSE POSITIVES PER PATHOLOGY (Factored in Metrics) ===")
                total_fp_in_metrics = df_results['False_Positives'].sum()
                total_predictions = df_results['Support'].sum()
                df_fp_sorted = df_results.sort_values(by='False_Positives', ascending=False)
                for _, row in df_fp_sorted.iterrows():
                    if row['False_Positives'] > 0:
                        fp_rate = (row['False_Positives'] / row['Support']) * 100 if row['Support'] > 0 else 0
                        print(f"{row['Pathology']:<20}: {int(row['False_Positives'])} false positives out of {int(row['Support'])} predictions ({fp_rate:.2f}%)")
                overall_fp_rate = (total_fp_in_metrics / total_predictions) * 100 if total_predictions > 0 else 0
                print(f"{'Total':<20}: {int(total_fp_in_metrics)} false positives out of {int(total_predictions)} predictions ({overall_fp_rate:.2f}%)")
                
                print("\n=== AVERAGE METRICS ===")
                print(f"Macro Precision: {np.nanmean(df_results['Precision']):.4f}")
                print(f"Macro Recall: {np.nanmean(df_results['Recall']):.4f}")
                print(f"Macro F1: {np.nanmean(df_results['F1']):.4f}")
                print(f"Macro AUROC: {np.nanmean(df_results['AUROC']):.4f}")
                print(f"Macro FPR: {np.nanmean(df_results['FPR']):.4f}")
                
                # Save false positives for fine-tuning (from comprehensive metrics - filtered for valid pathologies)
                fp_df = pd.DataFrame(false_positive_cases)
                fp_df.to_csv("false_positives_by_pathology_filtered.csv", index=False)
                print(f"\nSaved {len(fp_df)} false positive samples (from pathologies with valid metrics) to 'false_positives_by_pathology_filtered.csv'")
                
                # Generate visualizations for fine-tuning analysis
                print("\n=== GENERATING FINE-TUNING ANALYSIS GRAPHS ===")
                
                # 1. False Positive Distribution by Pathology (Bar Chart)
                plt.figure(figsize=(12, 6))
                pathology_fp_counts = df_results[df_results['False_Positives'] > 0].sort_values('False_Positives', ascending=True)
                plt.barh(pathology_fp_counts['Pathology'], pathology_fp_counts['False_Positives'])
                plt.xlabel('Number of False Positives')
                plt.ylabel('Pathology')
                plt.title('False Positive Distribution by Pathology\n(Target Classes for Fine-tuning)')
                plt.tight_layout()
                plt.savefig('false_positive_distribution_by_pathology.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Prediction Confidence Distribution of False Positives
                plt.figure(figsize=(12, 8))
                pathologies_with_fp = fp_df['pathology'].unique()
                n_plots = len(pathologies_with_fp)
                n_cols = 3
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                for i, pathology in enumerate(pathologies_with_fp):
                    plt.subplot(n_rows, n_cols, i+1)
                    pathology_fps = fp_df[fp_df['pathology'] == pathology]['prediction_prob']
                    plt.hist(pathology_fps, bins=20, alpha=0.7, edgecolor='black')
                    plt.xlabel('Prediction Confidence')
                    plt.ylabel('Frequency')
                    plt.title(f'{pathology}\n({len(pathology_fps)} FPs)')
                    plt.axvline(threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
                
                plt.tight_layout()
                plt.savefig('false_positive_confidence_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Performance vs False Positive Rate Scatter Plot
                plt.figure(figsize=(10, 8))
                # Calculate FP rate per pathology
                fp_rates = []
                aurocs = []
                pathology_names = []
                fp_counts = []
                
                for _, row in df_results.iterrows():
                    if row['False_Positives'] > 0:  # Only include pathologies with FPs
                        fp_rate = (row['False_Positives'] / row['Support']) * 100
                        fp_rates.append(fp_rate)
                        aurocs.append(row['AUROC'])
                        pathology_names.append(row['Pathology'])
                        fp_counts.append(row['False_Positives'])
                
                # Create scatter plot with bubble size based on FP count
                scatter = plt.scatter(fp_rates, aurocs, s=[x*10 for x in fp_counts], alpha=0.6, c=range(len(fp_rates)), cmap='viridis')
                plt.xlabel('False Positive Rate (%)')
                plt.ylabel('AUROC Score')
                plt.title('Model Performance vs False Positive Rate\n(Bubble size = # of False Positives)')
                
                # Add pathology labels
                for i, name in enumerate(pathology_names):
                    plt.annotate(name, (fp_rates[i], aurocs[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('performance_vs_false_positive_rate.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 4. Fine-tuning Priority Matrix (Heatmap)
                plt.figure(figsize=(10, 8))
                # Create priority scores: high FP count + low AUROC = high priority
                priority_data = []
                for _, row in df_results.iterrows():
                    if row['False_Positives'] > 0:
                        # Normalize scores (0-1) and combine: high FP count + low performance = high priority
                        fp_norm = row['False_Positives'] / df_results['False_Positives'].max()
                        auroc_norm = 1 - (row['AUROC'] if not np.isnan(row['AUROC']) else 0.5)  # Invert AUROC (low = bad)
                        priority_score = (fp_norm + auroc_norm) / 2
                        
                        priority_data.append({
                            'Pathology': row['Pathology'],
                            'FP_Count': row['False_Positives'],
                            'AUROC': row['AUROC'],
                            'Priority_Score': priority_score
                        })
                
                priority_df = pd.DataFrame(priority_data).sort_values('Priority_Score', ascending=False)
                
                # Create heatmap data
                heatmap_data = priority_df[['FP_Count', 'AUROC', 'Priority_Score']].T
                heatmap_data.columns = priority_df['Pathology']
                
                import seaborn as sns
                plt.figure(figsize=(12, 6))
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
                plt.title('Fine-tuning Priority Matrix\n(Higher Priority Score = More Critical for Fine-tuning)')
                plt.ylabel('Metrics')
                plt.xlabel('Pathologies')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('finetuning_priority_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 5. Class Imbalance Analysis for Fine-tuning
                plt.figure(figsize=(12, 6))
                positive_counts = []
                negative_counts = []
                pathology_labels = []
                
                for _, row in df_results.iterrows():
                    if row['False_Positives'] > 0:
                        # Estimate positive/negative distribution
                        total_samples = row['Support']
                        # Rough estimate based on typical medical dataset imbalance
                        estimated_positives = max(1, int(total_samples * 0.1))  # Assume ~10% positive rate
                        estimated_negatives = total_samples - estimated_positives
                        
                        positive_counts.append(estimated_positives)
                        negative_counts.append(estimated_negatives)
                        pathology_labels.append(row['Pathology'])
                
                x = np.arange(len(pathology_labels))
                width = 0.35
                
                plt.bar(x - width/2, negative_counts, width, label='Negative Samples', alpha=0.8)
                plt.bar(x + width/2, positive_counts, width, label='Positive Samples', alpha=0.8)
                
                plt.xlabel('Pathologies')
                plt.ylabel('Sample Count')
                plt.title('Class Distribution for Fine-tuning Target Pathologies\n(Consider data augmentation for minority classes)')
                plt.xticks(x, pathology_labels, rotation=45, ha='right')
                plt.legend()
                plt.yscale('log')  # Log scale due to imbalance
                plt.tight_layout()
                plt.savefig('class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("Generated fine-tuning analysis graphs:")
                print("  1. false_positive_distribution_by_pathology.png - Shows which pathologies need most attention")
                print("  2. false_positive_confidence_distribution.png - Shows confidence patterns of false positives")
                print("  3. performance_vs_false_positive_rate.png - Identifies problematic pathologies")
                print("  4. finetuning_priority_matrix.png - Priority ranking for fine-tuning")
                print("  5. class_imbalance_analysis.png - Class distribution analysis")
                
                if excluded_pathologies:
                    print(f"Excluded pathologies from CSV (no valid metrics): {excluded_pathologies}")
                    # Count false positives from excluded pathologies
                    excluded_fp_count = sum(1 for fp in false_positives if fp['pathology'] in excluded_pathologies)
                    print(f"False positives from excluded pathologies: {excluded_fp_count}")
        
        # Save ALL false positives (including those from skipped pathologies) for complete analysis
        if false_positives:
            all_fp_df = pd.DataFrame(false_positives)
            all_fp_df.to_csv("false_positives_all.csv", index=False)
            print(f"Saved ALL {len(all_fp_df)} false positive samples to 'false_positives_all.csv'")
            print(f"Difference: {len(all_fp_df) - len(fp_df) if len(cached_predictions) > 0 else len(all_fp_df)} false positives from pathologies excluded from metrics")
                
        else:
            print("No false positives found in the analyzed batches.")
            print("Consider:")
            print("1. Lowering the threshold (currently {})".format(threshold))
            print("2. Analyzing more batches")
            print("3. The model might be performing very well on this validation set")
        

except FileNotFoundError:
    print("\n--- Error ---")
    print("Could not find the dataset at the specified path.")
    print("Please ensure the `imgpath` and `csvpath` are correct.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("This might be due to an issue with file permissions or a missing file.")