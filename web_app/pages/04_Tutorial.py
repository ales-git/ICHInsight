import streamlit as st

video_file = open("/home/alessia/Documents/Projects/ICH_seg/interface/web_app/Tutorial.mp4", "rb")
video_bytes = video_file.read()

st.video(video_bytes) 