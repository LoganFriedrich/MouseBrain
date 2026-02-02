"""
quantification.py - Regional cell quantification for BrainSlice

Assigns cells to atlas regions and produces per-region counts with
positive/negative breakdown.

Usage:
    from mousebrain.pipeline_2d.sliceatlas.core.quantification import RegionQuantifier

    quantifier = RegionQuantifier(atlas_manager)
    cell_data = quantifier.assign_cells_to_regions(measurements, atlas_labels)
    region_counts = quantifier.count_per_region(cell_data)
    quantifier.export_results(cell_data, region_counts, output_dir, sample_id)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# Lazy imports
_pd = None


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


class RegionQuantifier:
    """
    Count cells per atlas region with positive/negative breakdown.
    """

    def __init__(self, atlas_manager: Optional[Any] = None):
        """
        Initialize quantifier.

        Args:
            atlas_manager: DualAtlasManager instance for region name lookup
                          If None, region names will be IDs only
        """
        self.atlas_manager = atlas_manager

    def assign_cells_to_regions(
        self,
        cell_measurements,  # DataFrame with centroid_y, centroid_x
        atlas_labels: np.ndarray,
        atlas_type: str = 'brain',
    ):
        """
        Assign each cell to an atlas region based on centroid location.

        Args:
            cell_measurements: DataFrame from ColocalizationAnalyzer
            atlas_labels: 2D array with region IDs per pixel
            atlas_type: 'brain' or 'spinal_cord' for region name lookup

        Returns:
            DataFrame with added columns:
            - region_id: atlas region ID at cell centroid
            - region_name: human-readable region name
        """
        pd = _get_pandas()
        df = cell_measurements.copy()

        region_ids = []
        region_names = []

        for _, row in df.iterrows():
            # Get pixel coordinates (ensure integer)
            y = int(round(row['centroid_y']))
            x = int(round(row['centroid_x']))

            # Clamp to image bounds
            y = max(0, min(y, atlas_labels.shape[0] - 1))
            x = max(0, min(x, atlas_labels.shape[1] - 1))

            # Get region ID at this location
            region_id = int(atlas_labels[y, x])
            region_ids.append(region_id)

            # Get region name
            if region_id == 0:
                region_names.append('Outside Atlas')
            elif self.atlas_manager is not None:
                name = self.atlas_manager.get_region_name(region_id, atlas_type)
                region_names.append(name)
            else:
                region_names.append(f'Region_{region_id}')

        df['region_id'] = region_ids
        df['region_name'] = region_names

        return df

    def count_per_region(
        self,
        cell_measurements,  # DataFrame with region_id, region_name, is_positive
    ):
        """
        Count positive/negative cells per region.

        Args:
            cell_measurements: DataFrame from assign_cells_to_regions()
                               Must have 'region_id', 'region_name', 'is_positive' columns

        Returns:
            DataFrame with columns:
            - region_id: atlas region ID
            - region_name: human-readable name
            - total_cells: total nuclei in region
            - positive_cells: nuclei classified as positive
            - negative_cells: nuclei classified as negative
            - positive_fraction: fraction of cells that are positive
        """
        pd = _get_pandas()
        df = cell_measurements

        # Check required columns
        required = ['region_id', 'region_name', 'is_positive']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Group by region
        grouped = df.groupby(['region_id', 'region_name']).agg({
            'label': 'count',  # Total cells
            'is_positive': 'sum'  # Positive cells
        }).reset_index()

        grouped.columns = ['region_id', 'region_name', 'total_cells', 'positive_cells']
        grouped['negative_cells'] = grouped['total_cells'] - grouped['positive_cells']
        grouped['positive_fraction'] = grouped['positive_cells'] / grouped['total_cells']

        # Sort by total cells descending
        grouped = grouped.sort_values('total_cells', ascending=False)

        return grouped

    def count_per_region_hierarchy(
        self,
        cell_measurements,  # DataFrame with region_id
        level: int = 1,
    ):
        """
        Count cells per region at a specific hierarchy level.

        Useful for aggregating to parent regions (e.g., all cortical areas).

        Args:
            cell_measurements: DataFrame with cell data
            level: Number of levels up in hierarchy (1 = immediate parent)

        Returns:
            DataFrame with counts per parent region
        """
        pd = _get_pandas()
        df = cell_measurements.copy()

        if self.atlas_manager is None:
            raise ValueError("Atlas manager required for hierarchy queries")

        # Get parent region for each cell
        parent_ids = []
        parent_names = []

        for region_id in df['region_id']:
            if region_id == 0:
                parent_ids.append(0)
                parent_names.append('Outside Atlas')
            else:
                parent_id = self.atlas_manager.get_parent_region(region_id, level=level)
                if parent_id is None:
                    parent_id = region_id  # Keep original if at top level
                parent_ids.append(parent_id)
                parent_names.append(self.atlas_manager.get_region_name(parent_id))

        df['parent_region_id'] = parent_ids
        df['parent_region_name'] = parent_names

        # Group by parent region
        grouped = df.groupby(['parent_region_id', 'parent_region_name']).agg({
            'label': 'count',
            'is_positive': 'sum'
        }).reset_index()

        grouped.columns = ['region_id', 'region_name', 'total_cells', 'positive_cells']
        grouped['negative_cells'] = grouped['total_cells'] - grouped['positive_cells']
        grouped['positive_fraction'] = grouped['positive_cells'] / grouped['total_cells']

        return grouped.sort_values('total_cells', ascending=False)

    def export_results(
        self,
        cell_data,  # DataFrame
        region_counts,  # DataFrame
        output_dir: Path,
        sample_id: str,
    ) -> Dict[str, Path]:
        """
        Export all results to CSV files.

        Args:
            cell_data: Per-cell measurements DataFrame
            region_counts: Per-region counts DataFrame
            output_dir: Directory for output files
            sample_id: Sample identifier for filenames

        Returns:
            Dict mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Per-cell measurements
        cell_path = output_dir / f"{sample_id}_cells.csv"
        cell_data.to_csv(cell_path, index=False)
        outputs['cells'] = cell_path

        # Per-region counts
        region_path = output_dir / f"{sample_id}_regions.csv"
        region_counts.to_csv(region_path, index=False)
        outputs['regions'] = region_path

        # Summary statistics
        summary = self.get_summary(cell_data, region_counts)
        summary_path = output_dir / f"{sample_id}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"BrainSlice Quantification Summary\n")
            f.write(f"Sample: {sample_id}\n")
            f.write(f"=" * 40 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        outputs['summary'] = summary_path

        return outputs

    def get_summary(
        self,
        cell_data,  # DataFrame
        region_counts,  # DataFrame
    ) -> Dict[str, Any]:
        """
        Get summary statistics from quantification results.

        Args:
            cell_data: Per-cell measurements
            region_counts: Per-region counts

        Returns:
            Dict with summary statistics
        """
        total_cells = len(cell_data)
        positive_cells = cell_data['is_positive'].sum() if 'is_positive' in cell_data.columns else 0
        n_regions = len(region_counts[region_counts['total_cells'] > 0])

        # Top region
        if len(region_counts) > 0:
            top_region = region_counts.iloc[0]
            top_region_name = top_region['region_name']
            top_region_count = top_region['total_cells']
        else:
            top_region_name = 'None'
            top_region_count = 0

        return {
            'total_cells': int(total_cells),
            'positive_cells': int(positive_cells),
            'negative_cells': int(total_cells - positive_cells),
            'positive_fraction': float(positive_cells / total_cells) if total_cells > 0 else 0.0,
            'regions_with_cells': int(n_regions),
            'top_region': top_region_name,
            'top_region_count': int(top_region_count),
        }


def quantify_sample(
    cell_measurements,  # DataFrame from colocalization
    atlas_labels: np.ndarray,
    atlas_manager: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    sample_id: str = 'sample',
) -> Tuple[Any, Any, Dict[str, Any]]:  # (cell_df, region_df, summary)
    """
    Convenience function for complete quantification.

    Args:
        cell_measurements: DataFrame from ColocalizationAnalyzer
        atlas_labels: 2D array with region IDs
        atlas_manager: Optional DualAtlasManager for region names
        output_dir: Optional directory for CSV export
        sample_id: Sample identifier

    Returns:
        Tuple of (cell_data, region_counts, summary)
    """
    quantifier = RegionQuantifier(atlas_manager)

    # Assign cells to regions
    cell_data = quantifier.assign_cells_to_regions(cell_measurements, atlas_labels)

    # Count per region
    region_counts = quantifier.count_per_region(cell_data)

    # Get summary
    summary = quantifier.get_summary(cell_data, region_counts)

    # Export if output directory provided
    if output_dir:
        quantifier.export_results(cell_data, region_counts, output_dir, sample_id)

    return cell_data, region_counts, summary
