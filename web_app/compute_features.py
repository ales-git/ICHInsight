import sys
import six
import re
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def print_features(features):
    for key, val in six.iteritems(features):
        print("\t%s: %s" % (key, val))


def compute_features(nii_image, adjusted_seg_data):
    #print("Received NII image path:", nii_image)
    #print("Received adjusted segmentation data:", adjusted_seg_data)

    import radiomics
    from radiomics import featureextractor

    #print("Radiomics version:", radiomics.__version__)

    # Check if the files exist or if the paths are correct
    if not (os.path.exists(nii_image) and os.path.exists(adjusted_seg_data)):
        print("Error: Input files not found.")
        return []

    # Instantiate extractor
    params = {'correctMask': True}
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    # Extract features
    features = extractor.execute(nii_image, adjusted_seg_data)
    features = {k: v for k, v in features.items() if not re.match(r'diagnostics_', k)}

    print_features(features)

    return features

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_features.py <nii_image_path> <adjusted_seg_data>")
        sys.exit(1)

    nii_image_path = sys.argv[1]
    adjusted_seg_data = sys.argv[2]

    compute_features(nii_image_path, adjusted_seg_data)
