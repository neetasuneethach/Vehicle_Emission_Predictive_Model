import cv2
from ultralytics import YOLO
import os
import streamlit as st
import tempfile
import time

def process_video(video_path, model_path, output_folder, total_parking_spaces):
    # Load the YOLO model
    model = YOLO(model_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the uploaded video file to a temporary location
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_path.read())

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return

    # Initialize variables
    frame_count = 0
    interval_count = 0
    prediction_interval = 5 * cap.get(cv2.CAP_PROP_FPS)  # Perform prediction every 10 seconds
    capture_interval = 5 * cap.get(cv2.CAP_PROP_FPS)  # Capture image every 20 seconds

    # Read the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Perform inference on the frame every 10-second interval
        if frame_count % prediction_interval == 0:
            results = model(frame)
            # You can further process the results here if needed

        # Capture image every 20 seconds
        if frame_count % capture_interval == 0:
            # Calculate total vehicles in the frame
            total_vehicles = 0
            if isinstance(results, list):
                for result in results:
                    if result.boxes:  # Check if boxes list is not empty
                        total_vehicles += len(result.boxes)
            else:
                if results.boxes:  # Check if boxes list is not empty
                    total_vehicles += len(results.boxes)

            # Calculate available parking spaces
            available_spaces = total_parking_spaces - total_vehicles

            # Print the total vehicles and available parking spaces
            interval_count += 1
            st.write(f"Interval {interval_count}: Total vehicles detected: {total_vehicles}, Available parking spaces: {available_spaces}")

            # Draw text overlay on the frame
            text = f"Total vehicles: {total_vehicles}, Available parking spaces: {available_spaces}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (16, 14, 16), 2)

            # Display the processed frame
            st.image(frame, channels="BGR")

            # Save the frame with drawn text overlay
            output_path = os.path.join(output_folder, f"interval_{interval_count}.jpg")
            cv2.imwrite(output_path, frame)

            # Pause execution for 1 second to display the frame
            time.sleep(1)

    # Release the video capture object
    cap.release()

def main():
    st.title("Real time Vehicle Detection and Parking Space Analytics")

    # Model and video file paths
    model_path = r'C:\Users\ChSuneetha\Desktop\Vehicle-Parkingdetection\yolov8s.pt'  # Update with your YOLOv8s model path & give the path of trained model
    video_path = st.file_uploader("Upload Video File", type=["mp4"])
    output_folder = r'C:\Users\ChSuneetha\Desktop\Vehicle-Parkingdetection\output'  # Update with your output folder path & give the path in which folder you want ot save the outputs

    # Define total parking spaces
    total_parking_spaces = st.number_input("Total Parking Spaces", min_value=1, step=1, value=100)

    if st.button("Process Video"):
        if video_path is None:
            st.error("Error: Please upload a video file.")
        else:
            process_video(video_path, model_path, output_folder, total_parking_spaces)

if __name__ == "__main__":
    main()
