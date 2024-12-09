import cv2
import numpy as np
import tensorflow as tf
import tempfile
import mediapipe as mp
from io import BytesIO
import utils.extraction as extraction
import utils.poseDetection as poseDetection
from utils.actions import action_fullname, all_actions

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # Convert TensorFlow tensor to NumPy scalar and display the probability
        cv2.putText(output_frame, action_fullname(actions[num]) + ': ' + str(round(prob.numpy(), 2)), 
                    (0, 185 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def ActionDetectionVideo(video_bytes, ViewProbabilities=False, ViewLandmarks=False, output_path='output_video.mp4'):
    actions = all_actions()
    model = tf.keras.models.load_model('mi_modelo.keras')
    model.summary()

    sequence = []
    sentence = []
    printed_sentence = []
    predictions = []
    threshold = 0.5
    score_left = 0  # Punto Izquierda
    score_right = 0  # Punto Derecha

    last_detected_action = None  # Track the last detected action

    # Create a temporary file for video input
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
        temp_input_path = temp_input_file.name
        temp_input_file.write(video_bytes.getvalue())

    input_video = cv2.VideoCapture(temp_input_path)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = input_video.read()
            if not ret:
                break
            
            image, results = poseDetection.mediapipe_detection(frame, holistic)
            if ViewLandmarks:
                poseDetection.draw_landmarks(image, results)
            
            keypoints = extraction.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                # Prepare the sequence for model prediction
                sequence_tensor = np.expand_dims(np.array(sequence), axis=0)

                res = model(sequence_tensor)[0]
                predicted_label = np.argmax(res)
                proba = res[predicted_label].numpy() 

                if proba > threshold: 
                    predictions.append(predicted_label)

                # Check if at least 15 out of the last 20 predictions are the same as the predicted label
                if predictions[-15:].count(predicted_label) >= 12:
                        if len(sentence) > 0: 
                            if actions[predicted_label] != sentence[-1]:
                                sentence.append(actions[predicted_label])               
                                printed_sentence.append(action_fullname(actions[predicted_label]))
                        else:
                            sentence.append(actions[predicted_label])
                            printed_sentence.append(action_fullname(actions[predicted_label]))
        
                # Increment score based on detected action (Punto Izquierda or Punto Derecha)
                current_action = printed_sentence[-1] if len(printed_sentence) > 0 else None

                # Check if the action is not the same as the last detected one
                if current_action != last_detected_action:
                    if current_action == "Punto Izquierda":
                        score_left += 1
                    elif current_action == "Punto Derecha":
                        score_right += 1

                    # Update the last detected action
                    last_detected_action = current_action
                
                
                if len(sentence) > 3: 
                    sentence = sentence[-3:]
                    printed_sentence = printed_sentence[-3:]

                if ViewProbabilities:
                    image = prob_viz(res, actions, image)
            
            # Draw a rectangle and text to show the score at the top center of the frame
            rect_x1, rect_x2 = (frame_width - 900) // 2, (frame_width + 900) // 2
            rect_y1, rect_y2 = 0, 50  # Position at the top of the frame
            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)
            
            # Display score in the format 'score_left : score_right'
            cv2.putText(image, f'{score_left} : {score_right}', 
                        (frame_width // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255, 255, 255), 2, cv2.LINE_AA)
            
            
            # Draw the sentence on the frame
            rect_x1, rect_x2 = (frame_width - 900) // 2, (frame_width + 900) // 2
            rect_y1, rect_y2 = frame_height - 110, frame_height -50
            text_size = cv2.getTextSize(' -> '.join(printed_sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame_width - text_size[0]) // 2

            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)
            cv2.putText(image, ' -> '.join(printed_sentence), (text_x, frame_height -70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Write the processed frame to the output video
            out.write(image)
            
        # Release the video and output objects
        input_video.release()
        out.release()
    # Return the output video file path
    return output_path

#def processing_video(uploaded_file_path, ViewProbabilities, ViewLandmarks, output_path='output_video.mp4'):
#    # Read the uploaded video file
#    with open(uploaded_file_path, 'rb') as f:
#        video_bytes = BytesIO(f.read())
#
#    # Process the video and save the output
#    processed_video_path = ActionDetectionVideo(video_bytes, ViewProbabilities, ViewLandmarks, output_path)
#    return processed_video_path
#
## File path for the uploaded video
#uploaded_file_path = '/home/soifa/repos/Computer-Vision-Project/prueba.MOV'
#output_path = '/home/soifa/repos/Computer-Vision-Project/processed_video.mp4'
#
## Process and save the video
#processed_video_path = processing_video(uploaded_file_path, True, True, output_path)
#
#print(f"Processed video saved at: {processed_video_path}")