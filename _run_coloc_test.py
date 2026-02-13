"""Temp runner - test coloc with vector contours."""
import sys, os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
print("Setting up...", flush=True)

sys.argv = [
    "run_coloc",
    r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_01_HD_Regions\E02_01_S13_DCN.nd2",
    "--dual",
    "--no-show",
    "--method", "zscore",
    "--split-touching",
]

from mousebrain.plugin_2d.sliceatlas.cli.run_coloc import main
print("Imports done, running pipeline...", flush=True)
main()
print("Done!", flush=True)
