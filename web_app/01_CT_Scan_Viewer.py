import streamlit as st
st.set_page_config(layout="wide")
import sys
# Add the parent directory to sys.path
sys.path.append('../')
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

image_size = 256

# Function to load or cache NII image
def load_or_cache_nii_image(uploaded_file):
    cache_key = f"nii_image_{uploaded_file.name}"
    if cache_key not in st.session_state:
        try:
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            nii_image = nib.load(temp_file_path)
            st.session_state[cache_key] = nii_image
        except Exception as e:
            st.error(f"Error loading NII file: {e}")
            return None
    return st.session_state[cache_key]

def main():
    st.title("CT Scan Viewer")
    st.sidebar.subheader("Tutorial")
    
    # File Upload
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload NII file", type=["nii", "nii.gz"])

    # Handle file deletion in /tmp directory
    if uploaded_file is None and "uploaded_nii" in st.session_state:
        st.sidebar.text(st.session_state.uploaded_nii.name + " has been deleted.")
        st.session_state.pop("uploaded_nii")
        return

    # Store uploaded image in session state
    if uploaded_file is not None:
        st.session_state.uploaded_nii = uploaded_file

    # Retrieve uploaded image from session state
    if "uploaded_nii" in st.session_state:
        uploaded_file = st.session_state.uploaded_nii
    
    ########################################################################
    
    def render_ai_segmentation_page():
        # Embed content of AI Segmentation page
        #st.title("AI Segmentation Page")
        st.switch_page("pages/02_AI_Segmentation.py")
    

    if uploaded_file is not None:
        # Display uploaded file
        st.sidebar.subheader("Uploaded NII file")
        st.sidebar.text(uploaded_file.name)

        try:
            # Load or cache NII file data
            nii_image = load_or_cache_nii_image(uploaded_file)
            original_image = nii_image

            #print(nii_image)

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
            slice_number = st.sidebar.slider("Select Slice", 0, nii_image.shape[2] - 1, nii_image.shape[2] // 2)
            slice_number_text = st.sidebar.number_input("Slice Number", value=slice_number, min_value=0, max_value=nii_image.shape[2] - 1)

            # Synchronize slider with text field
            slice_number = int(slice_number_text)
            
            # Display ROI selection
            st.sidebar.subheader("ROI Slice Selection")

            # Initialize first and last ROI slice numbers
            if "first_roi_slice_number" not in st.session_state:
                st.session_state.first_roi_slice_number = 0
            if "last_roi_slice_number" not in st.session_state:
                st.session_state.last_roi_slice_number = nii_image.shape[2] - 1
            

            side_col1, side_col2 = st.sidebar.columns([1, 1])

            # first slice ROI
            with side_col1:
                first_roi_slice_button = st.button("First ROI Slice")
                
                if first_roi_slice_button:
                    st.session_state.first_roi_slice_number = slice_number
                
                st.text(st.session_state.first_roi_slice_number)

            # last slice ROI
            with side_col2:
                last_roi_slice_button = st.button("Last ROI Slice")
                if last_roi_slice_button:
                    st.session_state.last_roi_slice_number = slice_number
                st.text(st.session_state.last_roi_slice_number)

            # Check if the last ROI slice number is greater than the first ROI slice number
            if st.session_state.last_roi_slice_number <= st.session_state.first_roi_slice_number:
                st.session_state.warning_message = "Last ROI Slice Number should be greater than the First ROI Slice Number."
                st.warning(st.session_state.warning_message)
            else:
                st.session_state.warning_message = None


            # Display AI segmentation button
            st.sidebar.subheader("AI Segmentation")
            if st.sidebar.button("AI Segmentation"):
                 # Reset segmentation flag
                st.session_state.segmentation_flag = False
                # Check if ROI coordinates are defined
                if "rectangle_coords" not in st.session_state or st.session_state.rectangle_coords is None:
                    st.session_state.warning_message = "ROI coordinates are not correctly defined. Please select and confirm a region of interest (ROI) on the axial slice before proceeding with AI Segmentation."
                    st.warning(st.session_state.warning_message)
                    time.sleep(3)
                else:
                    # Check if first and last ROI slice numbers are different
                    if st.session_state.first_roi_slice_number == st.session_state.last_roi_slice_number:
                        st.session_state.warning_message = "First and Last ROI Slice Numbers cannot be the same."
                    else:
                        print('rect coords before page2:',st.session_state.rectangle_coords)
                        st.session_state.selected_page = "AI Segmentation" # let's move to the dedicated app page
                        st.session_state.uploaded_image_content = nii_image
                        st.session_state.original_image = original_image
                        render_ai_segmentation_page() 
            

            # Display slices
            rectangle_coords = display_slices(nii_image, slice_number, min_clip, max_clip)
            if rectangle_coords is not None:
                # Save rectangle coordinates in a Python array
                rectangle_coords_array = np.array(rectangle_coords)
                # Now you can use rectangle_coords_array in your code as needed
        except Exception as e:
            print(f"Error loading NII file: {e}")
            #st.markdown("<h3 style='color:red; position: fixed; top: 0; width: 100%; text-align: center;'>Warning: Please select and confirm a region of interest (ROI) on the axial slice before proceeding with AI Segmentation.</h3>", unsafe_allow_html=True)
            st.warning(f"Please select and confirm a region of interest (ROI) on the axial slice before proceeding with AI Segmentation.")



def display_slices(nii_image, slice_number, min_clip, max_clip):
    # Extract slice data
    axial_slice_data = nii_image.get_fdata()[:, :, slice_number]
    sagittal_slice_data = np.rot90(nii_image.get_fdata()[:, slice_number, :])
    coronal_slice_data = np.rot90(nii_image.get_fdata()[slice_number, :, :])

    # Clip grayscale values between min_clip and max_clip
    axial_slice_data = np.clip(axial_slice_data, min_clip, max_clip)
    sagittal_slice_data = np.clip(sagittal_slice_data, min_clip, max_clip)
    coronal_slice_data = np.clip(coronal_slice_data, min_clip, max_clip)

    # Normalize voxel values to range [0.0, 1.0]
    axial_slice_data = (axial_slice_data - min_clip) / (max_clip - min_clip)
    sagittal_slice_data = (sagittal_slice_data - min_clip) / (max_clip - min_clip)
    coronal_slice_data = (coronal_slice_data - min_clip) / (max_clip - min_clip)

    #  Resize axial slice to original size
    axial_slice_original_size = axial_slice_data.copy()

    # Resize all slices of all orientations to match the aspect ratio of the original axial slice
    aspect_ratio = axial_slice_original_size.shape[1] / axial_slice_original_size.shape[0]
    target_height = 200
    target_width = int(target_height * aspect_ratio)
    axial_slice_data = cv2.resize(axial_slice_data, (target_width, target_height))
    sagittal_slice_data = cv2.resize(sagittal_slice_data, (target_width, target_height))
    coronal_slice_data = cv2.resize(coronal_slice_data, (target_width, target_height))

    # Create a placeholder for the canvas 
    # The first column takes N1 parts while the second 
    # column takes N2 parts of the total available width. 
    col1, col2 = st.columns([1, 3])

    # Display slices
    with col1:
        st.image([axial_slice_data, sagittal_slice_data, coronal_slice_data], 
                 caption=["Axial", "Sagittal", "Coronal"], 
                 width=200,
                 use_column_width=False, 
                 clamp=False,  # Disable clamping for correct visualization
                 channels="GRAY")  # Set channels to "GRAY" for grayscale display

    with col2:
        # Draw rectangle overlapped on the axial slice
        bg_image = Image.fromarray((axial_slice_data * 255).astype(np.uint8), mode='L')
        label_color = (
            st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
        )  

        # Define mode outside of if-else block
        mode = "transform"

        # Reactivate the draw ROI button when the trash bin icon is clicked
        if st.button("Select ROI"):
            mode = "rect"  # Only enable rectangle drawing when the button is clicked

        # Adding the Confirm ROI button
        if st.button("Confirm ROI"):
            canvas_result = st_canvas(
                fill_color=label_color,
                stroke_width=3,
                background_image=bg_image,
                height=600,
                width=600,
                drawing_mode=mode,
                key="color_annotation_app",
            )

            # Retrieve drawn rectangles
            df = pd.json_normalize(canvas_result.json_data["objects"])

            #print('df:',df)
            #print('len:',len(df))
            if len(df) > 1:
                st.session_state.warning_message = "You can only draw one rectangle at a time."
                st.warning(st.session_state.warning_message)
                # empty the rectangle coordinates
                st.session_state["rectangle_coords"] = None
                return None
            elif len(df) == 1:
                # Retrieve rectangle coordinates
                left = df.iloc[0]["left"]
                top = df.iloc[0]["top"]
                width = df.iloc[0]["width"]
                height = df.iloc[0]["height"]

                rectangle_coords = [left, top, left + width, top + height]

                #print('rect coords pre conversion:',rectangle_coords)

                # Convert coordinates to original dimensions
                print(axial_slice_original_size.shape)
                rectangle_coords[0] *= (256 / 600)
                rectangle_coords[1] *= (256/ 600)
                rectangle_coords[2] *= (256 / 600)
                rectangle_coords[3] *= (256 / 600)

                #print('rect coords post conversion:',rectangle_coords)

                # Store rectangle coordinates in session state
                st.session_state["rectangle_coords"] = rectangle_coords

                # Check if only one rectangle is present
                if len(df) == 1:
                    # Print coordinates and send them back to Streamlit
                    st.write("ROI Coordinates:", [round(r) for r in rectangle_coords])
                
                
        # Create the canvas
        canvas_result = st_canvas(
            fill_color=label_color,
            stroke_width=3,
            background_image=bg_image,
            height=600,
            width=600,
            drawing_mode=mode,
            key="color_annotation_app",
        )

        if canvas_result.json_data is not None and mode == "rect":
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if len(df) > 1:  # If more than one rectangle drawn
                st.session_state.warning_message = "You can only draw one rectangle at a time."
                st.warning(st.session_state.warning_message)
                # empty the rectangle coordinates
                st.session_state["rectangle_coords"] = None
                return None  # Return None to indicate no valid rectangle coordinates
            elif len(df) == 1:
                # Retrieve rectangle coordinates
                left = df.iloc[0]["left"]
                top = df.iloc[0]["top"]
                width = df.iloc[0]["width"]
                height = df.iloc[0]["height"]

                rectangle_coords = [left, top, left + width, top + height]

                #print('rect coords pre conversion:',rectangle_coords)

                # Convert coordinates to original dimensions
                print(axial_slice_original_size.shape)
                rectangle_coords[0] *= (256 / 600)
                rectangle_coords[1] *= (256/ 600)
                rectangle_coords[2] *= (256 / 600)
                rectangle_coords[3] *= (256 / 600)

                #print('rect coords post conversion:',rectangle_coords)

                # Store rectangle coordinates in session state
                st.session_state["rectangle_coords"] = rectangle_coords

                # Return rectangle coordinates
                return rectangle_coords


if __name__ == "__main__":
    main()

