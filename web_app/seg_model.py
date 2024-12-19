import sys
# Add the parent directory to sys.path
sys.path.append('../')
import threading
import numpy as np
from skimage import transform
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import matplotlib.pyplot as plt
import streamlit as st  # Import streamlit to enable slider
import nibabel as nib
import re
import os
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
from ultralytics import YOLO
import yaml
import csv
from pathlib import PosixPath
import io
from io import BytesIO
import tempfile
from config import webapp as config

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.3])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.3])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=1))

def crop_image(image, rect_coord):
    x_min, y_min, x_max, y_max = rect_coord
    print('image shape before cropping:',image.shape)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

def adjust_yolo_coords(yolo_coords, rect_coord):
    x_offset, y_offset = rect_coord[:2]
    adjusted_coords = []
    for box in yolo_coords:
        x_min, y_min, x_max, y_max = box
        adjusted_box = [x_min + x_offset, y_min + y_offset, x_max + x_offset, y_max + y_offset]
        adjusted_coords.append(adjusted_box)
    return adjusted_coords

def preprocess_ct(image_data, first_slice, last_slice, image_size=256):
    #%% set up the model
    device = 'cuda:0'   
    sam_model = sam_model_registry['vit_b'](checkpoint=config.medsam_weights_path).to(device)

    try:
        imgs = []
        img_embeddings = []
 
        #print(image_data.shape)

        # Preprocess image data
        lower_bound = 40
        upper_bound = 80
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        image_data_pre[image_data == 0] = 0
        image_data_pre = np.uint8(image_data_pre)

        z_min = int(first_slice)
        z_max = int(last_slice+1)

        # Ensure slice indices are within bounds
        #print('--')
        #print(image_data.shape[0])
        #print('zmin:',z_min)
        #print('zmax:', z_max)

        if z_min < 0 or z_min >= image_data.shape[0] or z_max < 0 or z_max > image_data.shape[0]+1:
            raise ValueError("Slice indices are out of bounds")

        for i in range(z_min, z_max):
            #print('image_data_pre shape:', image_data_pre.shape)
            img_slice_i = transform.resize(image_data_pre[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            # convert to three channels
            img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
            assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
            imgs.append(img_slice_i)
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(img_slice_i)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            # model input: (1, 3, 1024, 1024)
            input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
            assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
            # input_imgs.append(input_image.cpu().numpy()[0])
            with torch.no_grad():
                embedding = sam_model.image_encoder(input_image)
                img_embeddings.append(embedding.cpu().numpy()[0])
            
        return imgs, img_embeddings

    except Exception as e:
        print('Error in preprocessing:', e)

def model_predict(img_np, box_np, sam_trans, sam_model_tune, device='cuda:0'):
    try:
        H, W = img_np.shape[:2]
        #print('H:',H)
        #print('W: ', W)
        #print('img_np shape: ',img_np.shape)
        resize_img = sam_trans.apply_image(img_np)
        print('resize_img shape:', resize_img.shape)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device) 
        print('resize_img_tensor shape:', resize_img_tensor.shape)
        input_image = sam_model_tune.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024) 
        #print('input_image shape: ', input_image.shape)
        #print(input_image.shape)
        with torch.no_grad():
            image_embedding = sam_model_tune.image_encoder(input_image.to(device))  # (1, 256, 64, 64)
            # convert box to 1024x1024 grid
            box = sam_trans.apply_boxes(box_np, (H, W))
            box = box.astype(float)  # Example conversion to float
            #print('box shape:', box.shape)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            #print('box torch shape:', box_torch.shape)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            #print('box torch shape new:', box_torch.shape)
            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            #print('sparse embeddings shape:', sparse_embeddings.shape)
            #print('dense embeddings shape:', dense_embeddings.shape)

            medsam_seg_prob, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            #print('medsam_seg_prob shape:', medsam_seg_prob.shape)
            medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
            #print('medsam_seg_prob shape:', medsam_seg_prob.shape)
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            #print('medsam_seg_prob shape:', medsam_seg_prob.shape)
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            #print('medsam_seg shape:', medsam_seg.shape)
        return medsam_seg
        
    except Exception as e:
        print('Error in model prediction:', e)
        return None

def print_yolo_coordinates(yolo_results):
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy  # Extract coordinates (x1, y1, x2, y2)
            confidence = box.conf  # Extract confidence score
            class_id = box.cls  # Extract class id
            print(f'Coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence: {confidence}, Class ID: {class_id}')

def compute_yolo_boxes(ori_imgs, rect_coords):
    yolo_model = YOLO(config.yolo_weights_path)
    all_yolo_boxes = []
    print('Computing YOLO boxes for all slices...')
    for ori_img in ori_imgs:
        cropped_image = crop_image(ori_img, rect_coords)
        yolo_results = yolo_model(cropped_image, max_det=1)
        if len(yolo_results[0].boxes.xyxy.cpu().numpy()) > 0:
            yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            print('yolo_boxes orig:', yolo_boxes)
            adjusted_boxes = adjust_yolo_coords(yolo_boxes, rect_coords)
            print('yolo_boxes adjust:', adjusted_boxes)
            all_yolo_boxes.append(adjusted_boxes)
        else:
            all_yolo_boxes.append(None)
    return all_yolo_boxes

def impute_missing_boxes(all_yolo_boxes, rect_coords):
    """
    Impute missing bounding boxes using the closest slice's bounding box.
    If no bounding boxes are detected at all, use the user-provided coordinates.
    """
    last_valid_box = None
    
    # Find indices of missing boxes
    missing_indices = [i for i, box in enumerate(all_yolo_boxes) if box is None]
    
    if not missing_indices:
        return all_yolo_boxes

    for idx in missing_indices:
        # Find the closest valid bounding box
        closest_valid_index = min(
            (i for i in range(len(all_yolo_boxes)) if all_yolo_boxes[i] is not None),
            key=lambda i: abs(i - idx),
            default=None
        )
        if closest_valid_index is not None:
            all_yolo_boxes[idx] = all_yolo_boxes[closest_valid_index]

    # If all boxes are still None, use the user-provided coordinates
    if all(box is None for box in all_yolo_boxes):
        all_yolo_boxes = [rect_coords for _ in range(len(all_yolo_boxes))]

    return all_yolo_boxes


def segmentation(ori_imgs, rect_coords, sam_trans, sam_model_tune, device):
    sam_segs = []
    sam_bboxes = []
    
    all_yolo_boxes = compute_yolo_boxes(ori_imgs, rect_coords)
    #print('all_yolo_boxes:', all_yolo_boxes)
    impute_missing_boxes(all_yolo_boxes, rect_coords)
    #print('all_yolo_boxes after imputation:', all_yolo_boxes)
    
    for img_id, ori_img in enumerate(ori_imgs):

        bbox = all_yolo_boxes[img_id][0]
        # transform list to numpy array
        bbox = np.array(bbox)
        #print('bbox:', bbox)

        # 2. Apply segmentation model to segment the lesion
        seg_mask = model_predict(ori_img, bbox, sam_trans, sam_model_tune, device)
        sam_segs.append(seg_mask)
        sam_bboxes.append(bbox)
        

    # Stack sam_segs to 3D volume and save as nii.gz
    sam_segs_3d = np.stack(sam_segs, axis=0)
    ori_imgs_3d = np.stack(ori_imgs, axis=0)

    sam_segs_3d = nib.Nifti1Image(sam_segs_3d, np.eye(4))
    ori_imgs_3d = nib.Nifti1Image(ori_imgs_3d, np.eye(4))

    return ori_imgs_3d, sam_segs_3d, sam_bboxes


def overlay_segmentation(ori_img, seg_mask, alpha=0.6):
    
    # Convert the grayscale image to RGB
    print('max_ori_img:',ori_img.max())
    ori_img_rgb = Image.fromarray((ori_img * 255).astype(np.uint8)).convert('RGBA')

    # Create a pure red mask
    red_mask = np.zeros_like(seg_mask)
    red_mask[seg_mask > 0] = 255

    # Convert the red mask to an RGBA image with alpha channel
    red_mask_rgba = np.zeros((*red_mask.shape, 4), dtype=np.uint8)
    red_mask_rgba[:, :, 0] = red_mask  # Red channel
    red_mask_rgba[:, :, 3] = int(255 * alpha) # Alpha channel

    # Convert the red mask to PIL image
    red_mask_pil = Image.fromarray(red_mask_rgba, mode='RGBA')

    # Resize the red mask to match the size of the original image
    red_mask_resized = red_mask_pil.resize(ori_img_rgb.size, resample=Image.NEAREST)

    # Convert ori_img_rgb to RGBA mode
    ori_img_rgba = ori_img_rgb.convert('RGBA')

    # Composite the original image and the red mask
    overlaid_img = Image.alpha_composite(ori_img_rgba, red_mask_resized)

    return np.array(overlaid_img)

def adjust_segmentation_shape(seg_data, nii_image, total_slices, min_slice, max_slice):
    # Create a new array filled with zeros with the same shape as the original image
    new_seg_data = np.zeros((total_slices,) + seg_data.shape[1:], dtype=seg_data.dtype)

    # Copy segmentation data for slices within the specified range
    new_seg_data[min_slice:max_slice+1, :, :] = seg_data
    new_seg_data = np.moveaxis(new_seg_data, 0, -1)
    
    # resize the segmentation mask to the original nii_image shape
    new_seg_data = transform.resize(new_seg_data, (nii_image.shape[0], nii_image.shape[1], nii_image.shape[2]), order=0, preserve_range=True, mode='constant', anti_aliasing=False)

    return new_seg_data

def render_ai_radiomics_page():
    st.switch_page("pages/03_Radiomics.py")

def display_segmentation_results(ori_imgs, nii_image, original_image, nii_seg_data, sam_bboxes, image_data, min_slice, max_slice):
    
    print('nii_image shape:', original_image.shape)
    print(original_image.shape[2])

    total_slices = original_image.shape[-1] # from original scan
    nii_image = original_image.get_fdata()

    # Adjust segmentation shape (to be the same as the original scan)
    adjusted_seg_data = adjust_segmentation_shape(nii_seg_data.get_fdata(), original_image, total_slices, min_slice, max_slice)

    # Display image contrast adjustment
    st.sidebar.subheader("Image Contrast Adjustment")
    min_clip = st.sidebar.slider("Minimum Clip Value", 0.0, 100.0, 40.0)
    min_clip_text = st.sidebar.number_input("Minimum Clip Value", value=min_clip, min_value=0.0, max_value=100.0)
    max_clip = st.sidebar.slider("Maximum Clip Value", 0.0, 100.0, 80.0)
    max_clip_text = st.sidebar.number_input("Maximum Clip Value", value=max_clip, min_value=0.0, max_value=100.0)

    # Synchronize slider with text field
    min_clip = float(min_clip_text)
    max_clip = float(max_clip_text)

    # Display slice selection
    st.sidebar.subheader("Slice Selection")
    slice_number = st.sidebar.slider("Select Slice", 0, nii_image.shape[2] - 1, min_slice) # the slicer is put to the first segmented slice as default
    slice_number_text = st.sidebar.number_input("Slice Number", value=slice_number, min_value=0, max_value=nii_image.shape[2] - 1)

    # Synchronize slider with text field
    slice_number = int(slice_number_text)
            
    # Extract slice data
    axial_slice_data = nii_image[:, :, slice_number]
    
    # move the slice axis to the last dimension for the segmentation mask
    axial_slice_data_seg = adjusted_seg_data[:, :, slice_number] 

    # Clip grayscale values between min_clip and max_clip
    axial_slice_data = np.clip(axial_slice_data, min_clip, max_clip)

    # Normalize voxel values to range [0.0, 1.0]
    axial_slice_data = (axial_slice_data - min_clip) / (max_clip - min_clip)

    # Overlay segmentation mask on grayscale image
    overlaid_img = overlay_segmentation(axial_slice_data, axial_slice_data_seg)

    # Display the overlaid image
    st.image(overlaid_img, caption="Segmentation Overlay", use_column_width=False)

    # save adjusted segmentation mask in session state
    st.session_state.adjusted_seg_data = adjusted_seg_data

    # Download button for the adjusted segmentation mask
    def save_nifti(mask, affine=np.eye(4), file_name='mask.nii.gz'):
        nifti_image = nib.Nifti1Image(mask, affine)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file:
            nib.save(nifti_image, temp_file.name)
            temp_file_path = temp_file.name
        
        with open(temp_file_path, 'rb') as f:
            nifti_data = f.read()
        
        os.remove(temp_file_path)
        return nifti_data


    nifti_file_content = save_nifti(np.array(adjusted_seg_data))

    st.download_button(
        label="Download Mask as NIfTI",
        data=nifti_file_content,
        file_name="pred_mask.nii.gz",
        mime="application/octet-stream"
    )

    if st.button("Compute Radiomics Features"):
        render_ai_radiomics_page()
    