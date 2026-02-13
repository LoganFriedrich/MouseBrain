import sys
sys.path.insert(0, r"Y:\2_Connectome\Tissue\MouseBrain\src")
print("1. importing config...", flush=True)
from mousebrain.plugin_2d.sliceatlas.core import config
print("2. importing io...", flush=True)
from mousebrain.plugin_2d.sliceatlas.core import io
print("3. importing detection...", flush=True)
from mousebrain.plugin_2d.sliceatlas.core import detection
print("4. importing colocalization...", flush=True)
from mousebrain.plugin_2d.sliceatlas.core import colocalization
print("5. all done!", flush=True)
