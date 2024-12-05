import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf 
from tensorflow.keras.models import load_model
import utils.extraction as extraction
import utils.poseDetection as poseDetection
from utils.actions import action_fullname, all_actions

################################################
# Try the model in real time on your webcam
################################################

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # Display action and probability for each action
        cv2.putText(output_frame, action_fullname(actions[num]) + ': ' + str(round(prob.numpy(), 2)), 
                    (0, 185 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def main(ViewProbabilities=True, ViewLandmarks=True):
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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened(): 
            ret, frame = cap.read()
            if not ret:
                break

            image, results = poseDetection.mediapipe_detection(frame, holistic)
            
            if ViewLandmarks:
                poseDetection.draw_landmarks(image, results)  # Show landmarks on video
            
            # Extract keypoints from the detected pose
            keypoints = extraction.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep last 30 frames in sequence

            if len(sequence) == 30:
                # Prepare the sequence for model prediction
                sequence_tensor = np.expand_dims(np.array(sequence), axis=0)
               
                # Make a prediction for the current sequence
                res = model(sequence_tensor)[0]
                predicted_label = np.argmax(res)  # Get the index of the max probability
                proba = res[predicted_label].numpy()  # Get the probability of the predicted label

                if proba > threshold: 
                    predictions.append(predicted_label)

                # Ensure that the last 15 out of 20 predictions are consistent
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
                        print(f"Left score: {score_left}")
                    elif current_action == "Punto Derecha":
                        score_right += 1
                        print(f"Right score: {score_right}")

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
            
            # Display the sentence (actions detected so far)
            rect_x1, rect_x2 = (frame_width - 900) // 2, (frame_width + 900) // 2
            rect_y1, rect_y2 = frame_height - 110, frame_height -50
            text_size = cv2.getTextSize(' -> '.join(printed_sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame_width - text_size[0]) // 2

            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)
            cv2.putText(image, ' -> '.join(printed_sentence), (text_x, frame_height -70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print(printed_sentence)

            # Display the video feed
            cv2.imshow('OpenCV Feed', image)
            
            # Check if the score reaches 25 for any side and print results
            if score_left >= 25 or score_right >= 25:
                print("Felicidades!")
                print(f"Puntos: {score_left} : {score_right}")
                break

            # q to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(ViewProbabilities=True, ViewLandmarks=True)
