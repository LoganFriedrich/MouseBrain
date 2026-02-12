@echo off
"G:\Program_Files\Conda\envs\sliceatlas\python.exe" "y:\2_Connectome\Tissue\2D_Slices\Script_Tools\SliceAtlas\evaluate_pipeline.py" %* > "y:\2_Connectome\Tissue\2D_Slices\Script_Tools\SliceAtlas\eval_output.txt" 2>&1
echo EXIT CODE: %errorlevel% >> "y:\2_Connectome\Tissue\2D_Slices\Script_Tools\SliceAtlas\eval_output.txt"
