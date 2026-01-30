import tifffile
import numpy as np

brain = tifffile.imread(r'y:\2_Connectome\3_Nuclei_Detection\samples\353_CNT_01_02_1p625x_z4\3_Registered_Atlas\registered_brain.tiff')
bounds = tifffile.imread(r'y:\2_Connectome\3_Nuclei_Detection\samples\353_CNT_01_02_1p625x_z4\3_Registered_Atlas\boundaries.tiff')

print(f'Brain shape: {brain.shape}, dtype: {brain.dtype}, min: {brain.min()}, max: {brain.max()}')
print(f'Boundaries shape: {bounds.shape}, dtype: {bounds.dtype}, min: {bounds.min()}, max: {bounds.max()}')
print(f'Brain unique values: {len(np.unique(brain))} unique values')
print(f'Boundaries unique values: {len(np.unique(bounds))} unique values')
