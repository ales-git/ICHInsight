import subprocess
import os
import tempfile
import nibabel as nib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import io

def save_nifti(image, seg_data):
    tmp_dir = '../'  # Adjust this path as needed
    os.makedirs(tmp_dir, exist_ok=True)
    image_file = tempfile.NamedTemporaryFile(suffix=".nii", delete=False, dir=tmp_dir)
    seg_file = tempfile.NamedTemporaryFile(suffix=".nii", delete=False, dir=tmp_dir)
    seg_img = nib.Nifti1Image(seg_data, image.affine)
    nib.save(image, image_file.name)
    nib.save(seg_img, seg_file.name)
    return image_file.name, seg_file.name

def parse_features_output(output):
    features_dict = {}
    lines = output.split('\n')
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            features_dict[key] = value
    return features_dict

def compute_radiomic_features(nii_image, adjusted_seg_data):
   
    venv_path = '/'+os.path.join(*os.getcwd().split('/')[:-1])+'/venv_radiomics/radiomics'
    print(venv_path)
    #venv_path = '/home/alessia/Documents/Projects/ICHSeg/venv_radiomics/radiomics'
    python_interpreter = os.path.join(venv_path, 'bin', 'python')
    image_path, seg_path = save_nifti(nii_image, adjusted_seg_data)
    
    result = subprocess.run([python_interpreter, os.getcwd()+'/compute_features.py', image_path, seg_path], capture_output=True)

    if result.returncode == 0:
        features_output = result.stdout.decode('utf-8')
        features_dict = parse_features_output(features_output)
        return features_dict
    else:
        st.error("Error computing radiomic features.")
        return None

def plot_features(features_dict):
    features_dict = {k: float(v) for k, v in features_dict.items()}
    sorted_features = dict(sorted(features_dict.items(), key=lambda item: item[1], reverse=True))
    
    keys = list(sorted_features.keys())
    values = np.array(list(sorted_features.values()))
    sns.set_style("whitegrid")
    
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=values.min(), vmax=values.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    bar_colors = [cmap(norm(value)) for value in values]
    
    data = {
        'Feature': keys,
        'Original Value': values,
        'Color': bar_colors
    }
    
    plt.figure(figsize=(12, len(features_dict) * 0.5))
    barplot = sns.barplot(
        x='Original Value',
        y='Feature',
        data=data,
        palette=bar_colors,
        dodge=False,
        legend=False
    )
    
    x_ticks = plt.xticks()[0]
    x_labels = [f"{tick}" for tick in x_ticks]
    plt.xticks(ticks=x_ticks, labels=x_labels)
    
    plt.xlabel('Feature Value')
    plt.ylabel('Feature Name')
    plt.title('Radiomic Features')
    
    for index, value in enumerate(values):
        barplot.text(
            value, index,
            f'{value:.1f}',
            color='black',
            ha='left',
            va='center'
        )
    
    cbar_ax = plt.gca().inset_axes([1.05, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Original Value')
    
    st.pyplot(plt.gcf())

def main():
    st.title("Radiomics")
    
    # Add an Info button linking to the radiomics documentation
    st.sidebar.markdown("[Radiomics Documentation](https://pyradiomics.readthedocs.io/en/latest/)")
    
    if "nii_image" not in st.session_state:
        st.error("No NII image loaded. Please upload an NII file in the previous page.")
        return
    
    nii_image = st.session_state.nii_image
    
    if "radiomic_features" not in st.session_state:
        adjusted_seg_data = st.session_state.adjusted_seg_data
        with st.spinner("Computing radiomic features. This may take a while, please wait..."):
            computed_features = compute_radiomic_features(nii_image, adjusted_seg_data)
        if computed_features:
            st.session_state.radiomic_features = computed_features
    
    selected_feature = st.sidebar.radio("Select feature category:", ['firstorder', 'shape', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'])
    
    if "radiomic_features" in st.session_state:
        filtered_features = {k: v for k, v in st.session_state.radiomic_features.items() if selected_feature in k}
        plot_features(filtered_features)

        features = st.session_state.radiomic_features
        # convert features to a pandas DataFrame
        features_df = pd.DataFrame(features.items(), columns=['Feature', 'Value'])
        # create a CSV file in memory
        csv_bytes = io.StringIO()
        features_df.to_csv(csv_bytes, index=False)

        st.sidebar.download_button(
            label="Download as CSV",
            data=csv_bytes.getvalue(),
            file_name="radiomics_features.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
