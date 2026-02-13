import sys
print("Python:", sys.version, flush=True)
print("Importing mousebrain...", flush=True)
try:
    from mousebrain.plugin_2d.sliceatlas.cli.run_coloc import main
    print("Import OK", flush=True)
except Exception as e:
    print(f"Import error: {e}", flush=True)
