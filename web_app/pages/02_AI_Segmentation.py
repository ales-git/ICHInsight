import streamlit as st
import json
import sys
sys.path.append('../..')
import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import os
import cv2
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from skimage import transform
import time
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from seg_model import preprocess_ct, model_predict, display_segmentation_results, segmentation
from streamlit_drawable_canvas import st_canvas


def preprocess_and_segment(nii_image, first_slice, last_slice, rect_coords):
    try:
        print("Performing automatic segmentation...")
        image_data = np.transpose(nii_image.get_fdata(), (2, 0, 1)) 
        
        #print(image_data.shape)

        # Preprocessing CT scan
        with st.spinner("Preprocessing CT scan..."):
            start_time = time.time()
            imgs, img_embeddings = preprocess_ct(image_data, first_slice, last_slice)
            end_time = time.time()
            print("Preprocessing completed in {:.2f} seconds.".format(end_time - start_time))
            st.write("Preprocessing completed successfully!")

        # Segmentation
        with st.spinner("Segmenting CT scan..."):
            device = 'cuda:0'   
            box_np = np.array(rect_coords, dtype=np.float32) 
            # round the coordinates to the nearest integer
            box_np = np.round(box_np).astype(np.int32)
            print('Box np coordinates input rect:', box_np)
            sam_model_tune = sam_model_registry['vit_b'](checkpoint='/home/alessia/Documents/Projects/ICH_seg/work_dir/MEDSAM/exp_1/sam_model_best.pth').to(device)
            sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)
            start_time = time.time()
            img3d, seg3d, box_np = segmentation(imgs, box_np, sam_trans, sam_model_tune, device)
            end_time = time.time()
            print("Segmentation completed in {:.2f} seconds.".format(end_time - start_time))

            return img3d, seg3d, box_np, image_data

    except Exception as e:
        print("Error performing automatic segmentation: " + str(e))
        return None, None, None, None


def ai_segmentation_page(nii_image, original_image, first_slice, last_slice, rect_coords, segmentation_flag):
    st.title("AI Segmentation")
    #print(segmentation_flag)
    
    # Check if segmentation is needed
    if not segmentation_flag:
        img3d, seg3d, box_np, image_data = preprocess_and_segment(nii_image, first_slice, last_slice, rect_coords)
        if img3d is not None:
            st.session_state.img3d = img3d
            st.session_state.seg3d = seg3d
            st.session_state.nii_image = nii_image
            st.session_state.box_np = box_np
            st.session_state.image_data = image_data
            st.session_state.first_slice = first_slice
            st.session_state.last_slice = last_slice
            st.session_state.segmentation_flag = True
            st.session_state.original_image = original_image
            segmentation_flag = True
        else:
            # Failed to preprocess and segment
            return

    # Display segmentation results if available
    if segmentation_flag:
        st.write("Segmentation completed successfully!")
        imgs = st.session_state.img3d
        medsam_seg = st.session_state.seg3d
        box_np = st.session_state.box_np
        image_data = st.session_state.image_data
        first_slice = st.session_state.first_slice
        last_slice = st.session_state.last_slice
        nii_image = st.session_state.nii_image
        original_image = st.session_state.original_image
        print(image_data.shape)

        display_segmentation_results(imgs, nii_image, original_image, medsam_seg, box_np, image_data, first_slice, last_slice)
        #print(segmentation_flag)
    else:
        pass

def info():
    st.title("AI Segmentation Results") 
    st.write("This page will display the results of the AI segmentation.")
    st.write("Please select the CT scan and the region of interest (ROI) to segment in the previous page.")

if __name__ == "__main__":
    segmentation_flag = st.session_state.get("segmentation_flag", False)
    if 'uploaded_nii' in st.session_state and 'first_roi_slice_number' in st.session_state and 'last_roi_slice_number' in st.session_state and 'rectangle_coords' in st.session_state:
        ai_segmentation_page(st.session_state.uploaded_image_content, st.session_state.original_image, st.session_state.first_roi_slice_number, st.session_state.last_roi_slice_number, st.session_state.rectangle_coords, segmentation_flag)
    else:
        info()