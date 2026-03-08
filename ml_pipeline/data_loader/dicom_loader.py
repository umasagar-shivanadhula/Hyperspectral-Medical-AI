
import pydicom
import numpy as np

def load_dicom_cube(path):

    ds = pydicom.dcmread(path)
    image = ds.pixel_array

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    return image.astype(np.float32)
