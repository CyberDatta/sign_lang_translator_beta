import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

def mediapipe_detection(image,model):
    image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

#v1 code
actions=np.array(['week','monday','question'])
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('action.h5')
v1 = {'model': model, 'actions': actions}

#v2 code


# versions: v1,v2,v3
def singhara(frames_array,threshold,version):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #iterating through the array of 30 frames
        sequence=[]
        for frame in frames_array:
            #getting positions of keypoints
            image, results = mediapipe_detection(frame, holistic)
            
            #creating data structures to properly store keypoint values
            keypoints = extract_keypoints(results)
            
            #
            sequence.append(keypoints)
        res = (version['model']).predict(np.expand_dims(sequence, axis=0))[0]
        if res[np.argmax(res)] > threshold:
            output_gesture=version['actions'][np.argmax(res)]
        else:
            output_geture=None
        return output_gesture
    
if __name__=='__main__':
    cap = cv2.VideoCapture(0)

    # Initialize variables
    frame_count = 0
    max_frames = 30
    frames = []

    # Loop to capture frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame from webcam")
            break
        
        # Convert frame to grayscale (optional)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Append frame to list
        frames.append(frame_gray)
        
        # Display frame
        cv2.imshow('Webcam Feed', frame_gray)
        
        # Increment frame count
        frame_count += 1
        
        # Break the loop when required number of frames captured
        if frame_count >= max_frames:
            break
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert frames list to numpy array
    frames_array = np.array(frames)

    # Print shape of numpy array
    print("Shape of frames array:", frames_array.shape)
