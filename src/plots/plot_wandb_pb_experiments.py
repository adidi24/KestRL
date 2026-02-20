#!/usr/bin/env python3
"""
Script to plot PAC-Bayes experiment results with custom styling.

This script creates plots comparing certified_return and empirical_performance
metrics from PB-SAC experiments with different line styles and same color.

Usage:
    python plot_wandb_pb_experiments.py --groups "pbsac_improved_9_Hopper-v5" --hex-color "#1f77b4" --environment "Hopper-v5"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install numpy pandas matplotlib seaborn scipy tensorboard")
    sys.exit(1)


class PACBayesPlotter:
    """Creates W&B-style plots from downloaded tensorboard data."""

    def __init__(self, data_dir: str = "wandb_tensorboards"):
        """Initialize plotter with data directory."""
        self.data_dir = Path(data_dir)
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Set up matplotlib style for publication quality
        # For now, disable LaTeX due to missing packages
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        plt.rcParams['font.size'] = 12
        self.latex_enabled = False
        print("Using Times New Roman font. Run 'tlmgr install type1cm' to enable LaTeX.")

        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.bottom'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        sns.set_palette(self.color_palette)
    def clear_cache(self, group_name: Optional[str] = None):
        """Clear cached data files."""
        if group_name:
            group_path = self.data_dir / group_name
            if group_path.exists():
                for cache_file in group_path.glob("**/.cache_*.json"):
                    cache_file.unlink()
        else:
            for cache_file in self.data_dir.glob("**/.cache_*.json"):
                cache_file.unlink()

    def load_tensorboard_data(self, run_path: Path, metric: str) -> Optional[pd.DataFrame]:
        """
        Load specific metric data from tensorboard files with optimizations.

        Args:
            run_path: Path to the run directory
            metric: Metric name (e.g., 'rollout/episodic_return')

        Returns:
            DataFrame with columns ['step', 'value', 'wall_time'] or None
        """
        try:
            # Check cache first
            cache_file = run_path / f".cache_{metric.replace('/', '_')}.json"
            if cache_file.exists():
                try:
                    df = pd.read_json(cache_file)
                    return df
                except:
                    # If cache is corrupted, continue with loading
                    pass

            # Find tensorboard event files
            event_files = list(run_path.glob("**/events.out.tfevents.*"))
            if not event_files:
                # Try alternative patterns
                event_files = list(run_path.glob("**/*tfevents*"))

            if not event_files:
                print(f"No tensorboard files found in {run_path}")
                return None

            # Use the most recent event file
            event_file = sorted(event_files, key=os.path.getmtime)[-1]

            # Use direct event file parsing for faster loading
            from tensorflow.python.summary.summary_iterator import summary_iterator

            data = []
            try:
                for summary in summary_iterator(str(event_file)):
                    for value in summary.summary.value:
                        if value.tag == metric:
                            data.append({
                                'step': summary.step,
                                'value': float(value.simple_value),
                                'wall_time': summary.wall_time
                            })
            except Exception as e:
                # Fallback to EventAccumulator if direct parsing fails
                print(f"Direct parsing failed, using EventAccumulator fallback: {e}")
                size_guidance = {
                    'scalars': 0,
                    'images': 0,
                    'histograms': 0,
                    'tensors': 0,
                }
                ea = EventAccumulator(str(event_file), size_guidance=size_guidance)
                ea.Reload()

                if metric not in ea.Tags()['scalars']:
                    available_metrics = ea.Tags()['scalars']
                    print(f"Metric '{metric}' not found in {run_path.name}")
                    print(f"Available metrics: {available_metrics}")
                    return None

                scalar_events = ea.Scalars(metric)
                data = []
                for event in scalar_events:
                    data.append({
                        'step': event.step,
                        'value': event.value,
                        'wall_time': event.wall_time
                    })

            if not data:
                print(f"No data found for metric '{metric}' in {run_path.name}")
                return None

            df = pd.DataFrame(data)
            df = df.sort_values('step').reset_index(drop=True)

            # Cache the result
            try:
                df.to_json(cache_file)
            except:
                pass  # Ignore cache write errors

            return df

        except Exception as e:
            print(f"Error loading data from {run_path}: {e}")
            return None

    def load_group_data(self, group_name: str, metric: str) -> Dict[str, pd.DataFrame]:
        """
        Load data for all runs in a group.

        Args:
            group_name: Name of the W&B group
            metric: Metric to load

        Returns:
            Dictionary mapping run names to DataFrames
        """
        group_path = self.data_dir / group_name

        if not group_path.exists():
            print(f"Group directory not found: {group_path}")
            return {}

        group_data = {}
        run_dirs = [d for d in group_path.iterdir() if d.is_dir()]

        print(f"Loading data for group '{group_name}' ({len(run_dirs)} runs)...")

        for run_dir in run_dirs:
            print(f"  Loading {run_dir.name}...")

            # Load metadata
            metadata_file = run_dir / "metadata.json"
            run_metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    run_metadata = json.load(f)

            # Load tensorboard data with caching
            df = self.load_tensorboard_data(run_dir, metric)

            if df is not None and not df.empty:
                # Add metadata to dataframe
                df['run_name'] = run_metadata.get('name', run_dir.name)
                df['run_id'] = run_metadata.get('id', run_dir.name.split('_')[-1])
                df['group'] = group_name

                # Extract algorithm name from group
                algorithm = group_name.split('_')[0]
                if algorithm.lower() == 'pbsac':
                    algorithm = 'PB-SAC (Ours)'
                elif algorithm.lower() == 'sac':
                    algorithm = 'SAC (Baseline)'
                df['algorithm'] = algorithm
                group_data[run_dir.name] = df

        print(f"Successfully loaded {len(group_data)} runs for group '{group_name}'")
        return group_data

    def exponential_moving_average(self, data: np.ndarray, smoothing: float = 0.9) -> np.ndarray:
        """
        Compute exponential moving average (EMA) similar to W&B.

        Args:
            data: Input data array
            smoothing: Smoothing factor (0-1, higher = more smoothing)

        Returns:
            Smoothed data array
        """
        if len(data) == 0:
            return data

        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = smoothing * ema[i-1] + (1 - smoothing) * data[i]

        return ema

    def time_weighted_ema(self, values: np.ndarray, timestamps: np.ndarray,
                         smoothing: float = 0.9) -> np.ndarray:
        """
        Compute time-weighted exponential moving average as used in W&B.

        Args:
            values: Value array
            timestamps: Timestamp array (wall time or steps)
            smoothing: Smoothing factor (0-1, higher = more smoothing)

        Returns:
            Time-weighted smoothed data array
        """
        if len(values) == 0:
            return values

        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]

        for i in range(1, len(values)):
            # Time difference factor
            time_diff = timestamps[i] - timestamps[i-1]
            # Adaptive smoothing based on time difference
            adaptive_smooth = min(smoothing, 1.0 - np.exp(-time_diff / 1000.0))
            smoothed[i] = adaptive_smooth * smoothed[i-1] + (1 - adaptive_smooth) * values[i]

        return smoothed

    def apply_smoothing(self, df: pd.DataFrame, smoothing: float,
                       method: str = "ema") -> pd.DataFrame:
        """
        Apply smoothing to a dataframe based on the specified method.

        Args:
            df: DataFrame with 'step', 'value', 'wall_time' columns
            smoothing: Smoothing factor
            method: Smoothing method ('ema' or 'time_weighted_ema')

        Returns:
            DataFrame with smoothed values
        """
        if smoothing == 0 or len(df) < 2:
            return df

        df_smooth = df.copy()

        if method == "time_weighted_ema":
            df_smooth['value'] = self.time_weighted_ema(
                df['value'].values, df['wall_time'].values, smoothing
            )
        else:  # default to regular EMA
            df_smooth['value'] = self.exponential_moving_average(
                df['value'].values, smoothing
            )

        return df_smooth

    def compute_confidence_bands(self, group_data: Dict[str, pd.DataFrame],
                               metric_col: str = 'value') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean and confidence bands across runs.

        Args:
            group_data: Dictionary of run DataFrames
            metric_col: Column name for the metric values

        Returns:
            Tuple of (steps, mean, lower_bound, upper_bound)
        """
        if not group_data:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Find common step range
        all_steps = []
        for df in group_data.values():
            all_steps.extend(df['step'].tolist())

        min_step, max_step = min(all_steps), max(all_steps)

        # Create interpolation grid
        step_grid = np.linspace(min_step, max_step, num=min(1000, len(all_steps)))

        # Interpolate all runs to common grid
        interpolated_values = []
        for df in group_data.values():
            if len(df) > 1:
                interp_values = np.interp(step_grid, df['step'], df[metric_col])
                interpolated_values.append(interp_values)

        if not interpolated_values:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Compute statistics
        values_array = np.array(interpolated_values)
        mean_values = np.mean(values_array, axis=0)
        std_values = np.std(values_array, axis=0)

        # Confidence interval (mean ± std)
        lower_bound = mean_values - std_values
        upper_bound = mean_values + std_values

        return step_grid, mean_values, lower_bound, upper_bound

    def plot_grouped_comparison(self, groups: List[str], metric: str = "rollout/episodic_return",
                              smoothing: float = 0.0, smoothing_method: str = "ema",
                              show_raw: bool = True, raw_opacity: float = 0.1,
                              figsize: Tuple[int, int] = (12, 8), chart_title: Optional[str] = None,
                              environment: Optional[str] = None, font_size_add: float = 0.0,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create W&B-style comparison plot for multiple groups.

        Args:
            groups: List of group names to compare
            metric: Metric to plot (y-axis)
            smoothing: EMA smoothing factor (0 = no smoothing, 0.9 = heavy smoothing)
            smoothing_method: Smoothing method ('ema' or 'time_weighted_ema')
            show_raw: Whether to show raw data with low opacity
            raw_opacity: Opacity for raw data (0-1)
            figsize: Figure size tuple
            chart_title: Custom chart title (uses metric name if None)
            environment: Environment name to display as title
            font_size_add: Value to add to all font sizes (e.g., 2.0 adds 2pts to all fonts)
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        all_group_data = {}

        # Load data for all groups
        for group in groups:
            group_data = self.load_group_data(group, metric)
            if group_data:
                all_group_data[group] = group_data

        if not all_group_data:
            print("No data loaded for any groups!")
            return fig

        color_idx = 0

        for group_name, group_data in all_group_data.items():
            color = self.color_palette[color_idx % len(self.color_palette)]
            algorithm = group_name.split('_')[0]

            # Plot individual runs with low opacity if requested
            if show_raw:
                for run_name, df in group_data.items():
                    ax.plot(df['step'], df['value'],
                           color=color, alpha=raw_opacity, linewidth=0.8)

            # Apply smoothing to individual runs if requested
            smoothed_group_data = {}
            if smoothing > 0:
                for run_name, df in group_data.items():
                    smoothed_group_data[run_name] = self.apply_smoothing(df, smoothing, smoothing_method)
            else:
                smoothed_group_data = group_data

            # Compute and plot mean with confidence bands
            steps, mean_values, lower_bound, upper_bound = self.compute_confidence_bands(smoothed_group_data)

            if len(steps) > 0:
                # Plot mean line
                if algorithm.lower() == 'pbsac':
                    print("    Renaming algorithm to PB-SAC (Ours)")
                    algorithm = 'PB-SAC (Ours)'
                    label = f"{algorithm}"
                elif algorithm.lower() == 'sac':
                    print("    Renaming algorithm to SAC (Baseline)")
                    algorithm = 'SAC (Baseline)'
                    label = f"{algorithm}"
                else:
                    print(algorithm)
                    label = f"{algorithm.upper()}"
                ax.plot(steps, mean_values, color=color, linewidth=2.5, label=label)

                # Plot confidence band
                ax.fill_between(steps, lower_bound, upper_bound,
                               color=color, alpha=0.2)

            color_idx += 1

        # Styling with LaTeX if available, otherwise regular text
        base_font_size = 12 + font_size_add
        title_font_size = 12 + font_size_add
        legend_font_size = 6 + font_size_add

        if self.latex_enabled:
            ax.set_xlabel(r'Global Step', fontsize=base_font_size)
            ax.set_ylabel(metric.replace('/', r' ').replace('_', r' ').title(),
                         fontsize=base_font_size)
        else:
            ax.set_xlabel('Global Step', fontsize=base_font_size)
            ax.set_ylabel(metric.replace('/', ' ').replace('_', ' ').title(),
                         fontsize=base_font_size)

        # Display environment name as title if provided
        if environment:
            ax.set_title(environment, fontsize=title_font_size, fontweight='bold', pad=20)

        # Generate title for filename
        if chart_title:
            title_for_filename = chart_title
        elif environment:
            title_for_filename = environment
        else:
            # Extract environment name for title
            env_name = groups[0].split('_', 1)[1] if groups else "Environment"
            title_for_filename = f'{metric.replace("/", " ").replace("_", " ").title()} - {env_name}'
            if smoothing > 0:
                smoothing_type = "Time-weighted EMA" if smoothing_method == "time_weighted_ema" else "EMA"
                title_for_filename += f' ({smoothing_type}: {smoothing:.1f})'

        # Legend without shadow
        legend = ax.legend(frameon=True, fancybox=False, shadow=False,
                          fontsize=legend_font_size, loc='upper left')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_linewidth(0.5)

        # Remove default grid and keep only left and bottom spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # Add subtle horizontal grid lines at y-ticks only
        ax.yaxis.grid(True, linestyle='-', alpha=0.1, color='gray', linewidth=0.5)
        ax.set_axisbelow(True)  # Put grid behind data

        # Make plot start from y-axis
        ax.set_xlim(left=0)
        ax.margins(x=0)

        # Format axes with increased font size
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(2,2))
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax.tick_params(axis='both', which='major', labelsize=base_font_size-1)

        # Set font size for scientific notation offset text (the "1e6" label)
        ax.xaxis.offsetText.set_fontsize(base_font_size-1)
        ax.yaxis.offsetText.set_fontsize(base_font_size-1)

        plt.tight_layout()

        # Save if requested with title in filename
        if save_path:
            # Add title to filename if not already included
            if title_for_filename and title_for_filename not in save_path:
                # Clean title for filename
                clean_title = "".join(c for c in title_for_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_title = clean_title.replace(' ', '_')

                # Insert title before file extension
                save_path_parts = save_path.rsplit('.', 1)
                if len(save_path_parts) == 2:
                    save_path = f"{save_path_parts[0]}_{clean_title}.{save_path_parts[1]}"
                else:
                    save_path = f"{save_path}_{clean_title}"

            # Determine save format and settings
            file_extension = save_path.lower().split('.')[-1] if '.' in save_path else 'png'

            if file_extension == 'eps':
                # EPS format settings - EPS doesn't handle transparency well
                # Convert transparent elements to rasterized versions
                plt.savefig(save_path, format='eps', bbox_inches='tight',
                           facecolor='white', edgecolor='none',
                           rasterized=False, transparent=False)
            else:
                # PNG/PDF and other formats
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')

            print(f"Plot saved to: {save_path}")

        return fig

    def plot_pac_bayes_comparison(self, groups: List[str], metrics: Optional[List[str]] = None,
                                hex_color: str = "#1f77b4", smoothing: float = 0.0,
                                smoothing_method: str = "ema", figsize: Tuple[int, int] = (12, 8),
                                chart_title: Optional[str] = None, environment: Optional[str] = None,
                                font_size_add: float = 0.0, hide_xlabel: bool = False,
                                save_path: Optional[str] = None, width_ratio: float = 0.4) -> plt.Figure:
        """
        Create comparison plot with specified metrics using same color, different line styles.

        Args:
            groups: List of group names (PB-SAC groups)
            metrics: List of metrics to plot (defaults to PAC-Bayes metrics if None)
            hex_color: Hex color for all lines (e.g., "#1f77b4")
            smoothing: EMA smoothing factor
            smoothing_method: Smoothing method
            figsize: Figure size tuple
            chart_title: Custom chart title
            environment: Environment name to display as title
            font_size_add: Value to add to all font sizes
            hide_xlabel: Whether to hide the x-axis label (default: False)
            save_path: Optional path to save the plot
            width_ratio: Width ratio for proportional sizing (default: 0.4 for 40% width)

        Returns:
            Matplotlib figure object
        """
        # Calculate adjusted figure size based on width ratio
        # This helps maintain proportional heights when placed side-by-side in LaTeX
        base_width, base_height = figsize
        adjusted_width = base_width * width_ratio / 0.6  # Scale relative to 60% reference
        adjusted_figsize = (adjusted_width, base_height)

        fig, ax = plt.subplots(figsize=adjusted_figsize)

        # Use provided metrics or default to PAC-Bayes metrics
        if metrics is None:
            metrics = ['pac_bayes/certified_return', 'pac_bayes/empirical_performance']

        # Generate line styles and labels based on number of metrics
        line_styles = ['--', '-', '-.', ':']  # solid, dashed, dash-dot, dotted
        if len(metrics) > len(line_styles):
            # Cycle through line styles if more metrics than styles
            line_styles = line_styles * ((len(metrics) // len(line_styles)) + 1)

        # Generate clean labels from metric names
        metric_labels = []
        for metric in metrics:
            # Clean up metric names for labels
            if metric == 'pac_bayes/certified_return':
                # label = rf'Lower bound on $\underset{{\theta \sim \rho}}{{\mathbb{{E}}}}\left[-\mathcal{{L}}(\theta)\right]$'
                label = 'Certified Discounted Return'
            elif metric == 'pac_bayes/empirical_performance':
                # label = rf'Empirical Discounted Return $\underset{{\theta \sim \rho}}{{\mathbb{{E}}}}\left[-\hat{{\mathcal{{L}}}}_{{\mathfrak{{D}}}}(\theta)\right]$'
                label = 'Empirical Discounted Return'
            elif 'pac_bayes/' in metric:
                label = metric.replace('pac_bayes/', '').replace('_', ' ').title()
            else:
                label = metric.replace('/', ' ').replace('_', ' ').title()
            metric_labels.append(label)

        for metric, line_style, label in zip(metrics, line_styles, metric_labels):
            # Load data for all groups for this metric
            all_group_data = {}
            for group in groups:
                group_data = self.load_group_data(group, metric)
                if group_data:
                    all_group_data[group] = group_data

            if not all_group_data:
                print(f"No data loaded for metric '{metric}'!")
                continue

            # Combine all runs from all groups for this metric (no raw lines - only mean)
            combined_data = {}
            for group_data in all_group_data.values():
                combined_data.update(group_data)

            # Apply smoothing to individual runs if requested
            smoothed_data = {}
            if smoothing > 0:
                for run_name, df in combined_data.items():
                    smoothed_data[run_name] = self.apply_smoothing(df, smoothing, smoothing_method)
            else:
                smoothed_data = combined_data

            # Compute and plot mean with confidence bands
            steps, mean_values, lower_bound, upper_bound = self.compute_confidence_bands(smoothed_data)

            if len(steps) > 0:
                # Plot mean line with specified color and line style
                ax.plot(steps, mean_values, color=hex_color, linewidth=2.5,
                       linestyle=line_style, label=label)

                # Plot confidence band
                ax.fill_between(steps, lower_bound, upper_bound,
                               color=hex_color, alpha=0.2)

        # Styling
        base_font_size = 12 + font_size_add
        title_font_size = 12 + font_size_add
        legend_font_size = 6 + font_size_add

        if self.latex_enabled:
            if not hide_xlabel:
                ax.set_xlabel(r'Global Step', fontsize=base_font_size)
            ax.set_ylabel('Value', fontsize=base_font_size)
        else:
            if not hide_xlabel:
                ax.set_xlabel('Global Step', fontsize=base_font_size)
            ax.set_ylabel('Value', fontsize=base_font_size)

        # Display environment name as title if provided
        if environment:
            ax.set_title(environment, fontsize=title_font_size, fontweight='bold', pad=20)

        # Generate title for filename
        if chart_title:
            title_for_filename = chart_title
        elif environment:
            title_for_filename = environment
        else:
            title_for_filename = "PAC_Bayes_Comparison"

        # Legend without shadow
        legend = ax.legend(frameon=True, fancybox=False, shadow=False,
                          fontsize=legend_font_size, loc='lower right')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.5)
        legend.get_frame().set_linewidth(0.5)

        # Remove default grid and keep only left and bottom spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # Add subtle horizontal grid lines at y-ticks only
        ax.yaxis.grid(True, linestyle='-', alpha=0.1, color='gray', linewidth=0.5)
        ax.set_axisbelow(True)

        # Make plot start from y-axis
        ax.set_xlim(left=0)
        ax.margins(x=0)

        # Format axes
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(2,2))
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax.tick_params(axis='both', which='major', labelsize=base_font_size-1)
        ax.xaxis.offsetText.set_fontsize(base_font_size-1)
        ax.yaxis.offsetText.set_fontsize(base_font_size-1)

        plt.tight_layout()

        # Save if requested
        if save_path:
            if title_for_filename and title_for_filename not in save_path:
                clean_title = "".join(c for c in title_for_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_title = clean_title.replace(' ', '_')
                save_path_parts = save_path.rsplit('.', 1)
                if len(save_path_parts) == 2:
                    save_path = f"{save_path_parts[0]}_{clean_title}.{save_path_parts[1]}"
                else:
                    save_path = f"{save_path}_{clean_title}"

            file_extension = save_path.lower().split('.')[-1] if '.' in save_path else 'png'
            if file_extension == 'eps':
                plt.savefig(save_path, format='eps', bbox_inches='tight',
                           facecolor='white', edgecolor='none',
                           rasterized=False, transparent=False)
            else:
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            print(f"Plot saved to: {save_path}")

        return fig

    def list_available_groups(self) -> List[str]:
        """List all available groups in the data directory."""
        if not self.data_dir.exists():
            print(f"Data directory not found: {self.data_dir}")
            return []

        groups = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted(groups)

    def list_available_metrics(self, group_name: str) -> List[str]:
        """List available metrics for a specific group."""
        group_path = self.data_dir / group_name

        if not group_path.exists():
            return []

        # Get first run directory
        run_dirs = [d for d in group_path.iterdir() if d.is_dir()]
        if not run_dirs:
            return []

        run_path = run_dirs[0]

        try:
            # Find tensorboard event files
            event_files = list(run_path.glob("**/events.out.tfevents.*"))
            if not event_files:
                event_files = list(run_path.glob("**/*tfevents*"))

            if not event_files:
                return []

            event_file = sorted(event_files, key=os.path.getmtime)[-1]

            # Load and get available metrics
            ea = EventAccumulator(str(event_file))
            ea.Reload()

            return sorted(ea.Tags()['scalars'])

        except Exception as e:
            print(f"Error reading metrics from {run_path}: {e}")
            return []


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Plot PAC-Bayes experiment comparisons from downloaded tensorboard data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot default PAC-Bayes metrics with blue color
  python plot_wandb_pb_experiments.py --groups "pbsac_improved_9_Hopper-v5" --hex-color "#1f77b4" --environment "Hopper-v5"

  # Plot specific PAC-Bayes metrics
  python plot_wandb_pb_experiments.py --groups "pbsac_improved_9_Hopper-v5" --metrics "pac_bayes/certified_return,pac_bayes/empirical_performance" --hex-color "#d62728" --environment "Hopper-v5"

  # Plot different metrics with same color, different line styles
  python plot_wandb_pb_experiments.py --groups "pbsac_improved_9_Hopper-v5" --metrics "rollout/episodic_return,train/actor_loss" --hex-color "#2ca02c" --environment "Hopper-v5"

  # Single metric plot
  python plot_wandb_pb_experiments.py --groups "pbsac_improved_9_Hopper-v5" --metrics "pac_bayes/certified_return" --hex-color "#9467bd" --environment "Hopper-v5"

  # Save as EPS with larger fonts and smoothing
  python plot_wandb_pb_experiments.py --groups "pbsac_improved_9_Hopper-v5" --metrics "pac_bayes/certified_return,pac_bayes/empirical_performance" --hex-color "#1f77b4" --smooth 0.3 --font-size-add 3.0 --save "pac_bayes.eps" --environment "Hopper-v5"
        """
    )

    parser.add_argument(
        '--groups',
        required=True,
        help='Comma-separated list of PB-SAC group names (e.g., "pbsac_improved_9_Hopper-v5")'
    )

    parser.add_argument(
        '--hex-color',
        default='#1f77b4',
        help='Hex color for all lines (default: #1f77b4 - blue)'
    )

    parser.add_argument(
        '--metrics',
        help='Comma-separated list of metrics to plot (e.g., "pac_bayes/certified_return,pac_bayes/empirical_performance"). If not specified, defaults to PAC-Bayes metrics.'
    )

    parser.add_argument(
        '--data-dir',
        default='wandb_tensorboards',
        help='Directory containing downloaded tensorboard data (default: wandb_tensorboards)'
    )

    parser.add_argument(
        '--smooth',
        type=float,
        default=0.0,
        help='EMA smoothing factor 0-1 (0=no smoothing, 0.9=heavy smoothing, default: 0.0)'
    )

    parser.add_argument(
        '--smoothing-method',
        choices=['ema'],
        default='ema',
        help='Smoothing method: ema (default: ema)'
    )

    parser.add_argument(
        '--title',
        help='Custom chart title (auto-generated if not provided)'
    )

    parser.add_argument(
        '--environment',
        help='Environment name to display as title (e.g., "Hopper-v5")'
    )

    parser.add_argument(
        '--font-size-add',
        type=float,
        default=0.0,
        help='Value to add to all font sizes (e.g., 2.0 adds 2 points to all fonts, default: 0.0)'
    )

    parser.add_argument(
        '--hide-xlabel',
        action='store_true',
        help='Hide the x-axis label (keeps ticks and scientific notation)'
    )
    parser.add_argument(
        '--width-ratio',
        type=float,
        default=0.4,
        help='Width ratio for proportional sizing in side-by-side plots (default: 0.4 for 40%% width)'
    )

    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[16, 6],
        help='Figure size as width height (default: 12 8)'
    )

    parser.add_argument(
        '--save',
        help='Path to save the plot (supports .png, .pdf, .eps, .svg formats, e.g., "pac_bayes.eps")'
    )

    parser.add_argument(
        '--list-groups',
        action='store_true',
        help='List available groups and exit'
    )

    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached tensorboard data and exit'
    )


    args = parser.parse_args()

    try:
        # Initialize plotter
        plotter = PACBayesPlotter(data_dir=args.data_dir)

        if args.list_groups:
            print("Available groups:")
            groups = plotter.list_available_groups()
            for group in groups:
                print(f"  - {group}")
            return

        if args.clear_cache:
            print("Clearing cached data...")
            plotter.clear_cache()
            print("Cache cleared successfully")
            return

        # Parse groups
        groups = [g.strip() for g in args.groups.split(',')]

        # Parse metrics
        if args.metrics:
            metrics = [m.strip() for m in args.metrics.split(',')]
            if not metrics:
                print("Error: At least one metric must be specified")
                return
        else:
            metrics = None  # Will use default PAC-Bayes metrics

        # Validate smoothing parameter
        if not (0 <= args.smooth <= 1):
            print("Error: --smooth must be between 0 and 1")
            return

        # Validate hex color
        if not args.hex_color.startswith('#') or len(args.hex_color) != 7:
            print("Error: --hex-color must be in format #RRGGBB (e.g., #1f77b4)")
            return

        # Create plot
        if metrics:
            print(f"Creating plot for groups: {groups}")
            print(f"Metrics: {metrics}")
        else:
            print(f"Creating PAC-Bayes plot for groups: {groups}")
            print("Using default PAC-Bayes metrics: pac_bayes/certified_return, pac_bayes/empirical_performance")

        print(f"Using color: {args.hex_color}")
        if args.smooth > 0:
            print(f"EMA smoothing: {args.smooth}")

        fig = plotter.plot_pac_bayes_comparison(
            groups=groups,
            metrics=metrics,
            hex_color=args.hex_color,
            smoothing=args.smooth,
            smoothing_method=args.smoothing_method,
            figsize=tuple(args.figsize),
            chart_title=args.title,
            environment=args.environment,
            font_size_add=args.font_size_add,
            hide_xlabel=args.hide_xlabel,
            width_ratio=args.width_ratio,
            save_path=args.save
        )

        # Show plot
        plt.show()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()