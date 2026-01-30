"""BrainSlice core module - image processing and analysis."""

from .config import (
    BRAINSLICE_ROOT,
    DATA_DIR,
    TRACKER_CSV,
    MODELS_DIR,
    parse_sample_name,
    get_sample_dir,
    get_sample_subdir,
    SampleDirs,
)

from .io import (
    load_image,
    extract_channels,
    save_tiff,
    get_channel_info,
    guess_channel_roles,
    find_images_in_folder,
    load_folder,
    load_folder_lazy,
)

from .detection import (
    NucleiDetector,
    detect_nuclei,
    list_available_models,
)

from .colocalization import (
    ColocalizationAnalyzer,
    analyze_colocalization,
)

from .quantification import (
    RegionQuantifier,
    quantify_sample,
)

from .atlas_utils import (
    DualAtlasManager,
    list_available_atlases,
    download_atlas,
)

from .insets import (
    InsetManager,
    InsetInfo,
    parse_inset_name,
    find_matching_insets,
)

from .inset_detection import (
    InsetDetectionPipeline,
)

from .registration import (
    find_best_atlas_slice,
    register_to_atlas,
    apply_registration,
    compute_registration_quality,
)

from .preprocessing import (
    preprocess_for_registration,
    preprocess_atlas_slice,
    extract_brain_mask,
    downsample_for_registration,
    normalize_intensity,
)

from .boundaries import (
    extract_region_boundaries,
    warp_points,
    boundaries_to_napari_shapes,
    extract_major_boundaries,
    get_brain_outline,
)

from .deepslice_wrapper import (
    DeepSliceWrapper,
    is_deepslice_available,
    predict_single_slice_position,
)

from .elastix_registration import (
    register_slice_to_atlas,
    is_elastix_available,
    register_affine_only,
    apply_transform_to_labels,
    warp_atlas_to_image,
)

__all__ = [
    # Config
    'BRAINSLICE_ROOT', 'DATA_DIR', 'TRACKER_CSV', 'MODELS_DIR',
    'parse_sample_name', 'get_sample_dir', 'get_sample_subdir', 'SampleDirs',
    # IO
    'load_image', 'extract_channels', 'save_tiff', 'get_channel_info', 'guess_channel_roles',
    'find_images_in_folder', 'load_folder', 'load_folder_lazy',
    # Detection
    'NucleiDetector', 'detect_nuclei', 'list_available_models',
    # Colocalization
    'ColocalizationAnalyzer', 'analyze_colocalization',
    # Quantification
    'RegionQuantifier', 'quantify_sample',
    # Atlas
    'DualAtlasManager', 'list_available_atlases', 'download_atlas',
    # Insets
    'InsetManager', 'InsetInfo', 'parse_inset_name', 'find_matching_insets',
    'InsetDetectionPipeline',
    # Registration (legacy)
    'find_best_atlas_slice', 'register_to_atlas', 'apply_registration',
    'compute_registration_quality',
    # Preprocessing
    'preprocess_for_registration', 'preprocess_atlas_slice', 'extract_brain_mask',
    'downsample_for_registration', 'normalize_intensity',
    # Boundaries
    'extract_region_boundaries', 'warp_points', 'boundaries_to_napari_shapes',
    'extract_major_boundaries', 'get_brain_outline',
    # DeepSlice
    'DeepSliceWrapper', 'is_deepslice_available', 'predict_single_slice_position',
    # Elastix Registration
    'register_slice_to_atlas', 'is_elastix_available', 'register_affine_only',
    'apply_transform_to_labels', 'warp_atlas_to_image',
]
