import streamlit as st
from io import BytesIO
from backend import ActionDetectionVideo  # assuming backend contains the ActionDetectionVideo function

# Model Path
MODEL_PATH = 'mi_modelo.keras'

# Function to process video
def processing_video(uploaded_file, ViewProbabilities, ViewLandmarks):
    video_bytes = BytesIO(uploaded_file.read())
    processed_video = ActionDetectionVideo(video_bytes, ViewProbabilities, ViewLandmarks)
    return processed_video

# FRONTEND
header = st.container()

with header:
    st.title("Contador de puntos de voley.")
    
    # Add checkboxes to toggle options
    col1, col2 = st.columns(2)
    with col1:
        ViewProbabilities = st.checkbox("View Probabilities", value=True)
    with col2:
        ViewLandmarks = st.checkbox("View Landmarks", value=True)
    
    # File uploader for video files
    uploaded_file = st.file_uploader("Subi tu video...", type=["mp4", "avi", "mov"])

    # Initialize session state for processed video if not already
    if "processed_video" not in st.session_state:
        st.session_state.processed_video = None

    if uploaded_file:
        # Handle video processing
        if st.session_state.processed_video is None:
            status_placeholder = st.empty()
            status_placeholder.warning('Processing your video... (this may take a minute)')

            # Process and get the processed video as BytesIO
            st.session_state.processed_video = processing_video(uploaded_file, ViewProbabilities, ViewLandmarks)
            status_placeholder.success('Video processed successfully!')
        
        # Display the processed video
        st.video(st.session_state.processed_video)
