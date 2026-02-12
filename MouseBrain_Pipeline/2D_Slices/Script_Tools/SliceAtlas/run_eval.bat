@echo off
"G:\Program_Files\Conda\envs\sliceatlas\python.exe" "y:\LAB_ROOT\Tissue\2D_Slices\Script_Tools\SliceAtlas\evaluate_pipeline.py" %* > "y:\LAB_ROOT\Tissue\2D_Slices\Script_Tools\SliceAtlas\eval_output.txt" 2>&1
echo EXIT CODE: %errorlevel% >> "y:\LAB_ROOT\Tissue\2D_Slices\Script_Tools\SliceAtlas\eval_output.txt"
