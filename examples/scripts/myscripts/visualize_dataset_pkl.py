import argparse
import pickle
import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
import textwrap
import logging
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

"""
Usage:
python3 examples/scripts/myscripts/visualize_dataset_pkl.py \
    --pkl_path /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCabToCounter/overlay_images_binary/dataset_cache_val_1__4_8_12_16__True_True.pkl \
    --output_dir temp

"""

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

TASK_DESC_TO_SYSTEM_PROMPT = {
    "PnPCounterToCab": "Pick the object from the counter and place it in the cabinet",
    "PnPCabToCounter": "Pick the object from the cabinet and place it on the counter",
    "PnPCounterToMicrowave": "Pick the object from the plate on the counter and place it in the microwave",
    "PnPMicrowaveToCounter": "Pick the object from the microwave and place it on the plate on the counter",
    "PnPStoveToCounter": "Pick the object from the stove and place it on the plate on the counter",  
    "PnPCounterToStove": "Pick the object from the plate on the counter and place it on the stove",  
    "PnPCounterToSink": "Pick the object from the plate on the counter and place it in the sink",  
    "PnPSinkToCounter": "Pick the object from the sink and place it on the plate on the counter",
    "PnPCoffeeServeMug": "Pick the mug from under the coffee machine dispenser and place it on the counter",
    "PnPCloseDrawer": "Close the drawer",
}

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset from a pickle file.")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to the dataset pickle file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations.")
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        logger.error(f"Pickle file not found: {pkl_path}")
        return

    logger.info(f"Loading dataset from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        combined_data = pickle.load(f)
    
    logger.info(f"Loaded {len(combined_data)} samples.")

    viz_dir = Path(args.output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving visualizations to: {viz_dir}")

    # ============================================================================================
    # VISUALIZATION LOGIC
    # ============================================================================================

    # 1. Basic Statistics
    logger.info(f"\n📊 Dataset Statistics:")
    logger.info(f"  Total number of samples: {len(combined_data)}")

    # Extract answer distribution
    answers = [item['messages'][2]['content'] for item in combined_data]
    answers_numeric = [int(a) for a in answers]

    logger.info(f"\n📈 Answer Distribution:")
    logger.info(f"  Mean answer: {np.mean(answers_numeric):.2f}")
    logger.info(f"  Median answer: {np.median(answers_numeric):.2f}")
    logger.info(f"  Std dev: {np.std(answers_numeric):.2f}")
    logger.info(f"  Min answer: {min(answers_numeric)}")
    logger.info(f"  Max answer: {max(answers_numeric)}")

    # Count positive vs negative answers
    positive_count = sum(1 for a in answers_numeric if a > 0)
    negative_count = sum(1 for a in answers_numeric if a < 0)
    zero_count = sum(1 for a in answers_numeric if a == 0)

    logger.info(f"\n📊 Answer Polarity:")
    logger.info(f"  Positive answers (right image shows more progress): {positive_count} ({positive_count/len(answers_numeric)*100:.1f}%)")
    logger.info(f"  Negative answers (right image shows less progress): {negative_count} ({negative_count/len(answers_numeric)*100:.1f}%)")
    logger.info(f"  Zero answers (same progress): {zero_count} ({zero_count/len(answers_numeric)*100:.1f}%)")

    # Demo type distribution
    demo_types = [item.get('demo_success', 'unknown') for item in combined_data]
    demo_type_counts = Counter(demo_types)
    logger.info(f"\n📁 Demo Type Distribution:")
    for demo_type, count in demo_type_counts.items():
        logger.info(f"  {demo_type}: {count} ({count/len(demo_types)*100:.1f}%)")

    # 2. Show some random examples
    logger.info(f"\n🖼️  Sample Examples (5 random samples):")
    logger.info("-"*80)

    random.seed(42)
    sample_indices = random.sample(range(len(combined_data)), min(5, len(combined_data)))

    for i, idx in enumerate(sample_indices, 1):
        item = combined_data[idx]
        demo_success = item.get('demo_success', 'unknown')
        success_label = 'SUCCESS' if demo_success == 'success' else 'FAILURE' if demo_success == 'failure' else 'UNKNOWN'
        logger.info(f"\nExample {i} (Index {idx}) - {success_label}:")
        logger.info(f"  Overlay Image: {item['images'][0]}")
        logger.info(f"  Original Image 1: {item['original_images'][0]}")
        logger.info(f"  Original Image 2: {item['original_images'][1]}")
        logger.info(f"  Demo Type: {item.get('demo_success', 'unknown')}")
        logger.info(f"  Demo Success: {demo_success}")
        logger.info(f"  Answer (progress difference): {item['messages'][2]['content']}")
        logger.info(f"  System Prompt: {item['messages'][0]['content'][:100]}...")
        logger.info(f"  User Question: {item['messages'][1]['content'][:100]}...")

    # 3. Create visualizations
    try:
        # Create comprehensive statistics plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dataset Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Answer distribution histogram
        ax = axes[0, 0]
        ax.hist(answers_numeric, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (Equal Progress)')
        ax.set_xlabel('Progress Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Progress Values\n(n={len(answers_numeric)})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Answer polarity pie chart
        ax = axes[0, 1]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        labels = [f'Positive\n({positive_count})', f'Negative\n({negative_count})', f'Zero\n({zero_count})']
        sizes = [positive_count, negative_count, zero_count]
        # Filter out zero sizes
        sizes_filtered = [s for s in sizes if s > 0]
        labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
        colors_filtered = [c for c, s in zip(colors, sizes) if s > 0]
        if sizes_filtered:
            ax.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered,
                  autopct='%1.1f%%', startangle=90)
        ax.set_title('Answer Polarity Distribution')

        # Plot 3: Progress value box plot
        ax = axes[0, 2]
        ax.boxplot(answers_numeric, vert=True)
        ax.set_ylabel('Progress Value')
        ax.set_title('Progress Value Distribution (Box Plot)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Positive vs Negative distribution bar chart
        ax = axes[1, 0]
        ax.bar(['Negative\n(L > R)', 'Zero\n(L = R)', 'Positive\n(R > L)'],
              [negative_count, zero_count, positive_count],
              color=['red', 'gray', 'green'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Progress Direction Distribution')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 5: Demo type breakdown
        ax = axes[1, 1]
        demo_type_names = list(demo_type_counts.keys())
        demo_type_values = list(demo_type_counts.values())
        ax.bar(demo_type_names, demo_type_values, alpha=0.7, edgecolor='black', color='purple')
        ax.set_xlabel('Demo Type')
        ax.set_ylabel('Count')
        ax.set_title('Pairs per Demo Type')
        ax.grid(True, alpha=0.3, axis='y')

        # Hide the last subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        stats_plot_path = viz_dir / 'combined_data_statistics.png'
        plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"\n📊 Saved statistics plot to: {stats_plot_path}")

        # Plot example images grid
        logger.info(f"\n🖼️  Creating visual grids of example images (Success & Failure)...")
        
        # Split data
        success_items_viz = [item for item in combined_data if item.get('demo_success') == 'success']
        failure_items_viz = [item for item in combined_data if item.get('demo_success') == 'failure']

        for viz_name, items_list in [('success', success_items_viz), ('failure', failure_items_viz)]:
            num_random = min(12, len(items_list))  # Show up to 12 random examples
            
            if num_random > 0:
                random_indices = random.sample(range(len(items_list)), num_random)
                rows = (num_random + 2) // 3  # 3 columns
                cols = min(3, num_random)

                fig, axes_grid = plt.subplots(rows, cols, figsize=(18, 6 * rows))
                fig.suptitle(f'Random {viz_name.capitalize()} Examples', fontsize=16, fontweight='bold')

                if rows == 1 and cols == 1:
                    axes_grid = [[axes_grid]]
                elif rows == 1:
                    axes_grid = [axes_grid]
                elif cols == 1:
                    axes_grid = [[ax] for ax in axes_grid]

                for idx, data_idx in enumerate(random_indices):
                    row = idx // 3
                    col = idx % 3

                    if row >= len(axes_grid) or col >= len(axes_grid[row]):
                        continue

                    ax = axes_grid[row][col]
                    item = items_list[data_idx]
                    overlay_path = item['images'][0]
                    answer = item['messages'][2]['content']
                    demo_success = item.get('demo_success', 'unknown')

                    try:
                        img = Image.open(overlay_path)
                        ax.imshow(img)
                        ax.axis('off')

                        progress_val = int(answer)
                        color = 'green' if progress_val > 0 else 'red' if progress_val < 0 else 'gray'
                        # Add success/failure label to title
                        success_label = '✓ Success' if demo_success == 'success' else '✗ Failure' if demo_success == 'failure' else 'Unknown'
                        
                        # Add task description
                        job_name = item.get('job_name', '')
                        task_desc = ""
                        for task_key, task_description in TASK_DESC_TO_SYSTEM_PROMPT.items():
                            if task_key in job_name:
                                task_desc = task_description
                                break
                        
                        # Wrap task description for display
                        if task_desc:
                            task_desc_short = textwrap.shorten(task_desc, width=40, placeholder="...")
                            title_text = f'Progress: {answer} ({success_label})\n{task_desc_short}'
                        else:
                            title_text = f'Progress: {answer} ({success_label})'

                        ax.set_title(title_text,
                                   fontsize=8, fontweight='bold', color=color)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error: {str(e)}',
                               ha='center', va='center', transform=ax.transAxes, fontsize=8)
                        ax.axis('off')

                # Hide any unused subplots
                for idx in range(num_random, rows * cols):
                    row = idx // 3
                    col = idx % 3
                    if row < len(axes_grid) and col < len(axes_grid[row]):
                        axes_grid[row][col].axis('off')

                plt.tight_layout()
                grid_path = viz_dir / f'{viz_name}_examples_grid.png'
                plt.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close()
                logger.info(f"🖼️  Saved {viz_name} example images grid to: {grid_path}")
            else:
                logger.info(f"No {viz_name} examples found to visualize.")

    except Exception as e:
        logger.warning(f"Failed to create visualizations: {e}")

    # Analyze demo_id and demo_id_exact statistics
    logger.info(f"\n📋 Demo ID Analysis:")

    # Extract demo_id and demo_id_exact from original_images paths
    demo_id_to_exact = defaultdict(set)
    demo_id_to_pairs = defaultdict(int)
    demo_id_exact_to_pairs = defaultdict(int)

    # NEW: Track success vs failure breakdown per demo_id
    demo_id_to_success_pairs = defaultdict(int)
    demo_id_to_failure_pairs = defaultdict(int)
    demo_id_exact_to_demo_type = {}

    for item in combined_data:
        # Extract demo_id and demo_id_exact from the original image paths
        # Format: .../demo_id_exact/frame_XXXXXX.png
        orig_img1 = item['original_images'][0]
        # Get the demo_id_exact from path (second to last part)
        demo_id_exact = Path(orig_img1).parent.name
        # Get the demo_id (first part before underscore)
        demo_id = demo_id_exact.split('_')[0]

        # Get demo type (success/failure)
        demo_type = item.get('demo_success', 'unknown')

        demo_id_to_exact[demo_id].add(demo_id_exact)
        demo_id_to_pairs[demo_id] += 1
        demo_id_exact_to_pairs[demo_id_exact] += 1
        demo_id_exact_to_demo_type[demo_id_exact] = demo_type

        # Track success vs failure per demo_id
        if demo_type == 'success':
            demo_id_to_success_pairs[demo_id] += 1
        elif demo_type == 'failure':
            demo_id_to_failure_pairs[demo_id] += 1

    # Log statistics
    logger.info(f"  Total unique demo_id: {len(demo_id_to_exact)}")
    logger.info(f"  Total unique demo_id_exact: {sum(len(exacts) for exacts in demo_id_to_exact.values())}")

    logger.info(f"\n📊 Demo ID breakdown (top 20 by pair count):")
    sorted_demo_ids = sorted(demo_id_to_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
    for demo_id, pair_count in sorted_demo_ids:
        num_exact = len(demo_id_to_exact[demo_id])
        success_count = demo_id_to_success_pairs[demo_id]
        failure_count = demo_id_to_failure_pairs[demo_id]
        logger.info(f"  demo_id '{demo_id}': {num_exact} unique demo_id_exact, {pair_count} image pairs (Success: {success_count}, Failure: {failure_count})")

    logger.info(f"\n📊 Demo ID Exact breakdown (top 20 by pair count):")
    sorted_demo_id_exact = sorted(demo_id_exact_to_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
    for demo_id_exact, pair_count in sorted_demo_id_exact:
        logger.info(f"  demo_id_exact '{demo_id_exact}': {pair_count} image pairs")

    # Create visualization for demo_id statistics
    try:
        # Create demo_id analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Demo ID Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Number of demo_id_exact per demo_id (bar graph)
        ax = axes[0, 0]
        # Sort demo_ids by number of demo_id_exact (descending)
        demo_id_sorted = sorted(demo_id_to_exact.items(), key=lambda x: len(x[1]), reverse=True)
        demo_ids_plot1 = [item[0] for item in demo_id_sorted]
        num_exact_counts = [len(item[1]) for item in demo_id_sorted]

        # Create bar graph
        x_pos = range(len(demo_ids_plot1))
        ax.bar(x_pos, num_exact_counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('demo_id')
        ax.set_ylabel('Number of demo_id_exact')
        ax.set_title(f'Number of demo_id_exact per demo_id\n(Total demo_id: {len(demo_id_to_exact)})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(demo_ids_plot1, rotation=90, fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean line
        mean_val = np.mean(num_exact_counts)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.legend()

        # Plot 2: Number of pairs per demo_id (top 20)
        ax = axes[0, 1]
        top_20_demo_ids = sorted_demo_ids[:20]
        demo_ids = [item[0] for item in top_20_demo_ids]
        pair_counts = [item[1] for item in top_20_demo_ids]
        ax.barh(range(len(demo_ids)), pair_counts, color='coral', edgecolor='black')
        ax.set_yticks(range(len(demo_ids)))
        ax.set_yticklabels(demo_ids, fontsize=8)
        ax.set_xlabel('Number of image pairs')
        ax.set_title('Top 20 demo_id by pair count')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 3: Number of pairs per demo_id_exact (top 20)
        ax = axes[1, 0]
        top_20_demo_id_exact = sorted_demo_id_exact[:20]
        demo_id_exacts = [item[0] for item in top_20_demo_id_exact]
        pair_counts_exact = [item[1] for item in top_20_demo_id_exact]
        ax.barh(range(len(demo_id_exacts)), pair_counts_exact, color='mediumseagreen', edgecolor='black')
        ax.set_yticks(range(len(demo_id_exacts)))
        ax.set_yticklabels(demo_id_exacts, fontsize=7)
        ax.set_xlabel('Number of image pairs')
        ax.set_title('Top 20 demo_id_exact by pair count')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 4: Scatter plot - demo_id_exact count vs pair count per demo_id
        ax = axes[1, 1]
        demo_ids_all = list(demo_id_to_exact.keys())
        x_vals = [len(demo_id_to_exact[did]) for did in demo_ids_all]
        y_vals = [demo_id_to_pairs[did] for did in demo_ids_all]
        ax.scatter(x_vals, y_vals, alpha=0.6, s=50, color='purple', edgecolor='black')
        ax.set_xlabel('Number of demo_id_exact per demo_id')
        ax.set_ylabel('Number of image pairs per demo_id')
        ax.set_title('Relationship: demo_id_exact count vs pair count')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(x_vals) > 1:
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save to both locations
        demo_analysis_path = viz_dir / 'demo_id_analysis.png'
        plt.savefig(demo_analysis_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"\n📊 Saved demo_id analysis plot to: {demo_analysis_path}")

    except Exception as e:
        logger.warning(f"Failed to create demo_id analysis visualizations: {e}")

    # Create SUCCESS vs FAILURE comparison visualization
    try:
        logger.info(f"\n📊 Creating Success vs Failure comparison visualization...")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Success vs Failure Data Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Stacked bar chart - Success vs Failure per demo_id (top 20)
        ax = axes[0, 0]
        top_20_demo_ids = sorted_demo_ids[:20]
        demo_ids = [item[0] for item in top_20_demo_ids]
        success_counts = [demo_id_to_success_pairs[did] for did in demo_ids]
        failure_counts = [demo_id_to_failure_pairs[did] for did in demo_ids]

        x = range(len(demo_ids))
        ax.barh(x, success_counts, label='Success', color='#2ecc71', edgecolor='black')
        ax.barh(x, failure_counts, left=success_counts, label='Failure', color='#e74c3c', edgecolor='black')
        ax.set_yticks(x)
        ax.set_yticklabels(demo_ids, fontsize=8)
        ax.set_xlabel('Number of image pairs')
        ax.set_title('Success vs Failure Pairs per demo_id (Top 20)')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 2: Grouped bar chart - Success vs Failure comparison (top 15)
        ax = axes[0, 1]
        top_15_demo_ids = sorted_demo_ids[:15]
        demo_ids_15 = [item[0] for item in top_15_demo_ids]
        success_counts_15 = [demo_id_to_success_pairs[did] for did in demo_ids_15]
        failure_counts_15 = [demo_id_to_failure_pairs[did] for did in demo_ids_15]

        x = np.arange(len(demo_ids_15))
        width = 0.35
        ax.bar(x - width/2, success_counts_15, width, label='Success', color='#2ecc71', edgecolor='black')
        ax.bar(x + width/2, failure_counts_15, width, label='Failure', color='#e74c3c', edgecolor='black')
        ax.set_xlabel('demo_id')
        ax.set_ylabel('Number of image pairs')
        ax.set_title('Success vs Failure Comparison (Top 15)')
        ax.set_xticks(x)
        ax.set_xticklabels(demo_ids_15, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Scatter plot - Success count vs Failure count per demo_id
        ax = axes[1, 0]
        all_demo_ids = list(demo_id_to_exact.keys())
        success_vals = [demo_id_to_success_pairs[did] for did in all_demo_ids]
        failure_vals = [demo_id_to_failure_pairs[did] for did in all_demo_ids]

        ax.scatter(success_vals, failure_vals, alpha=0.6, s=80, color='purple', edgecolor='black')
        ax.set_xlabel('Number of Success pairs')
        ax.set_ylabel('Number of Failure pairs')
        ax.set_title('Success vs Failure Pairs Distribution')
        ax.grid(True, alpha=0.3)

        # Add diagonal reference line (equal success/failure)
        max_val = max(max(success_vals) if success_vals else 0, max(failure_vals) if failure_vals else 0)
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Equal success/failure')
        ax.legend()

        # Plot 4: Pie chart - Overall Success vs Failure distribution
        ax = axes[1, 1]
        total_success = sum(demo_id_to_success_pairs.values())
        total_failure = sum(demo_id_to_failure_pairs.values())

        sizes = [total_success, total_failure]
        labels = [f'Success\n({total_success} pairs)', f'Failure\n({total_failure} pairs)']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        if sum(sizes) > 0:
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax.set_title(f'Overall Success vs Failure Distribution\n(Total: {sum(sizes)} pairs)')

        plt.tight_layout()

        # Save to both locations
        success_failure_plot = viz_dir / 'success_vs_failure_comparison.png'
        plt.savefig(success_failure_plot, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"\n📊 Saved success vs failure comparison plot to: {success_failure_plot}")

        # Log summary statistics
        logger.info(f"\n📊 Success vs Failure Summary:")
        logger.info(f"  Total Success pairs: {total_success} ({total_success/(total_success+total_failure)*100:.1f}%)")
        logger.info(f"  Total Failure pairs: {total_failure} ({total_failure/(total_success+total_failure)*100:.1f}%)")
        logger.info(f"  demo_ids with only Success data: {sum(1 for did in all_demo_ids if demo_id_to_success_pairs[did] > 0 and demo_id_to_failure_pairs[did] == 0)}")
        logger.info(f"  demo_ids with only Failure data: {sum(1 for did in all_demo_ids if demo_id_to_failure_pairs[did] > 0 and demo_id_to_success_pairs[did] == 0)}")
        logger.info(f"  demo_ids with both Success and Failure data: {sum(1 for did in all_demo_ids if demo_id_to_success_pairs[did] > 0 and demo_id_to_failure_pairs[did] > 0)}")

    except Exception as e:
        logger.warning(f"Failed to create success vs failure comparison visualization: {e}")

    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()
