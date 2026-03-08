
import multiprocessing as mp
import numpy as np

def _process_patch(args):
    patch, spectral_fn, spatial_fn = args
    spectral = spectral_fn(patch)
    spatial = spatial_fn(patch)
    return np.concatenate([spectral, spatial])

def extract_features_parallel(patches, spectral_fn, spatial_fn, workers=None):
    if workers is None:
        workers = mp.cpu_count()

    with mp.Pool(workers) as pool:
        features = pool.map(
            _process_patch,
            [(p, spectral_fn, spatial_fn) for p in patches]
        )

    return np.array(features)
