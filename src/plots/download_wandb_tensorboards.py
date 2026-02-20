#!/usr/bin/env python3
"""
Custom script to download TensorBoard logs from Weights & Biases grouped experiments.

This script retrieves all runs from a specified W&B group and downloads their
tensorboard data to a local directory structure. Optimized for large files
and includes error handling for network issues.

Usage:
    python download_wandb_tensorboards.py --group "pbsac_HalfCheetah-v4"
    python download_wandb_tensorboards.py --group "sac_Ant-v4" --entity "your_entity"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import time

try:
    import wandb
    from tqdm import tqdm
    import requests
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install wandb tqdm requests")
    sys.exit(1)


class WandbTensorboardDownloader:
    """Downloads tensorboard data from W&B runs in a group."""

    def __init__(self, entity: Optional[str] = None, project: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            entity: W&B entity (will use from environment if not provided)
            project: W&B project (will use from environment if not provided)
        """
        # Load environment variables from .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Initialize wandb
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            print("Warning: WANDB_API_KEY not found in environment")

        self.entity = entity or os.getenv('WANDB_ENTITY')
        self.project = project or os.getenv('WANDB_PROJECT')

        if not self.entity or not self.project:
            print("Error: Entity and project must be provided via arguments or environment variables")
            print("Set WANDB_ENTITY and WANDB_PROJECT in your environment or .env file")
            sys.exit(1)

        self.api = wandb.Api()
        print(f"Initialized W&B client for {self.entity}/{self.project}")

    def get_runs_in_group(self, group_name: str) -> List:
        """
        Get all runs in a specified group.

        Args:
            group_name: Name of the W&B group

        Returns:
            List of wandb Run objects
        """
        print(f"Fetching runs from group: {group_name}")

        try:
            # Query runs with the specified group
            runs = self.api.runs(f"{self.entity}/{self.project}",
                               filters={"group": group_name})

            run_list = list(runs)
            print(f"Found {len(run_list)} runs in group '{group_name}'")

            if len(run_list) == 0:
                print(f"No runs found in group '{group_name}'. Available groups:")
                self._list_available_groups()

            return run_list

        except Exception as e:
            print(f"Error fetching runs: {e}")
            return []

    def _list_available_groups(self, limit: int = 10):
        """List available groups in the project."""
        try:
            all_runs = self.api.runs(f"{self.entity}/{self.project}")
            groups = set()

            for run in all_runs:
                if hasattr(run, 'group') and run.group:
                    groups.add(run.group)
                if len(groups) >= limit:
                    break

            print(f"Available groups (showing first {limit}):")
            for group in sorted(groups):
                print(f"  - {group}")

        except Exception as e:
            print(f"Error listing groups: {e}")

    def download_tensorboard_data(self, run, output_dir: Path,
                                max_retries: int = 3) -> bool:
        """
        Download tensorboard data for a single run.

        Args:
            run: wandb Run object
            output_dir: Directory to save the tensorboard data
            max_retries: Maximum number of download retries

        Returns:
            True if successful, False otherwise
        """
        run_dir = output_dir / f"{run.name}_{run.id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing run: {run.name} (ID: {run.id})")

        # Save run metadata
        metadata = {
            'name': run.name,
            'id': run.id,
            'state': run.state,
            'created_at': str(run.created_at),
            'config': dict(run.config),
            'summary': dict(run.summary),
            'tags': run.tags,
            'notes': run.notes,
            'group': run.group,
            'url': run.url
        }

        with open(run_dir / 'metadata.json', 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)

        # Try to download tensorboard files
        tensorboard_files = []

        try:
            # Get all files for this run
            files = run.files()

            # Filter for tensorboard-related files
            for file in files:
                if any(pattern in file.name.lower() for pattern in
                      ['tfevents', 'tensorboard', 'events.out', 'tb_logs']):
                    tensorboard_files.append(file)

            if not tensorboard_files:
                print(f"  No tensorboard files found for run {run.name}")
                # Try to download any log files as fallback
                for file in files:
                    if any(pattern in file.name.lower() for pattern in
                          ['log', 'events', 'scalar']):
                        tensorboard_files.append(file)

            if not tensorboard_files:
                print(f"  No relevant files found for run {run.name}")
                return False

            print(f"  Found {len(tensorboard_files)} tensorboard-related files")

            # Download each file with progress bar and retry logic
            for file in tqdm(tensorboard_files, desc=f"Downloading {run.name}"):
                success = False
                for attempt in range(max_retries):
                    try:
                        file_path = run_dir / file.name
                        file_path.parent.mkdir(parents=True, exist_ok=True)

                        # Download with streaming for large files
                        if hasattr(file, 'download'):
                            file.download(root=str(run_dir), replace=True)
                        else:
                            # Fallback method
                            url = file.url
                            self._download_file_with_progress(url, file_path)

                        success = True
                        break

                    except Exception as e:
                        print(f"    Attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff

                if not success:
                    print(f"    Failed to download {file.name} after {max_retries} attempts")

            print(f"  Completed download for run {run.name}")
            return True

        except Exception as e:
            print(f"  Error downloading run {run.name}: {e}")
            return False

    def _download_file_with_progress(self, url: str, file_path: Path,
                                   chunk_size: int = 8192):
        """Download a file with progress tracking."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(file_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             desc=file_path.name) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise

    def download_group_tensorboards(self, group_name: str,
                                  output_dir: str = "wandb_tensorboards"):
        """
        Download tensorboard data for all runs in a group.

        Args:
            group_name: Name of the W&B group
            output_dir: Directory to save all the data
        """
        output_path = Path(output_dir) / group_name
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_path.absolute()}")

        # Get all runs in the group
        runs = self.get_runs_in_group(group_name)

        if not runs:
            return

        # Download tensorboard data for each run
        successful_downloads = 0
        failed_downloads = 0

        print(f"\nStarting download of {len(runs)} runs...")

        for i, run in enumerate(runs, 1):
            print(f"\n[{i}/{len(runs)}] Processing run: {run.name}")

            try:
                success = self.download_tensorboard_data(run, output_path)
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1

            except KeyboardInterrupt:
                print("\nDownload interrupted by user")
                break
            except Exception as e:
                print(f"Unexpected error processing run {run.name}: {e}")
                failed_downloads += 1

        print(f"\n{'='*50}")
        print(f"Download Summary:")
        print(f"  Successful: {successful_downloads}")
        print(f"  Failed: {failed_downloads}")
        print(f"  Total: {len(runs)}")
        print(f"  Output directory: {output_path.absolute()}")
        print(f"{'='*50}")

        # Create a summary file
        summary_file = output_path / "download_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Download Summary for group: {group_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Entity: {self.entity}\n")
            f.write(f"Project: {self.project}\n")
            f.write(f"Successful downloads: {successful_downloads}\n")
            f.write(f"Failed downloads: {failed_downloads}\n")
            f.write(f"Total runs: {len(runs)}\n\n")

            f.write("Downloaded runs:\n")
            for run in runs:
                f.write(f"  - {run.name} ({run.id}): {run.url}\n")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Download TensorBoard logs from W&B grouped experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all runs from a specific group
  python download_wandb_tensorboards.py --group "pbsac_HalfCheetah-v4"

  # Specify entity and project explicitly
  python download_wandb_tensorboards.py --group "sac_Ant-v4" --entity "your_entity" --project "your_project"

  # Custom output directory
  python download_wandb_tensorboards.py --group "pbac_Walker2d-v4" --output "./my_tensorboards"
        """
    )

    parser.add_argument(
        '--group',
        required=True,
        help='W&B group name (e.g., "pbsac_HalfCheetah-v4")'
    )

    parser.add_argument(
        '--entity',
        help='W&B entity (uses WANDB_ENTITY env var if not provided)'
    )

    parser.add_argument(
        '--project',
        help='W&B project (uses WANDB_PROJECT env var if not provided)'
    )

    parser.add_argument(
        '--output', '-o',
        default='wandb_tensorboards',
        help='Output directory (default: wandb_tensorboards)'
    )

    parser.add_argument(
        '--list-groups',
        action='store_true',
        help='List available groups and exit'
    )

    args = parser.parse_args()

    try:
        # Initialize downloader
        downloader = WandbTensorboardDownloader(
            entity=args.entity,
            project=args.project
        )

        if args.list_groups:
            print("Listing available groups...")
            downloader._list_available_groups(limit=50)
            return

        # Download tensorboard data
        downloader.download_group_tensorboards(
            group_name=args.group,
            output_dir=args.output
        )

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()