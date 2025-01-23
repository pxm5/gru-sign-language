import mediapipe as mp
import cv2 as cv
import numpy as np
import json
import torch
import model

def process(files) -> list:
    detections = []
    mpHands = mp.solutions.hands

    # All default Parameters for now
    hands = mpHands.Hands(
        static_image_mode = False, 
        max_num_hands=2, 
        model_complexity = 1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
        )
    
    for file in files:
        cap = cv.VideoCapture(file)
        video_detections = []
        while True:
            
            ret, frame = cap.read()

            if not ret:
                break
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if not results.multi_hand_landmarks: continue

            for hand_landmarks in results.multi_hand_landmarks:
                output = []
                for id, landmark in enumerate(hand_landmarks.landmark):
                    output.append([id, landmark.x, landmark.y, landmark.z])
                video_detections.append(output)
        
        # video_detections = video_detections
        detections.append(video_detections)

    return detections


def getVideos(json_path, n_classes = 20) -> list[str]:

    with open(json_path, 'r') as file:
        data = json.load(file)
        output = []
        x = 0
        for class_ in data:
            if x == n_classes: break
            class_output = [class_['gloss']]
            temp = []
            for instance in class_["instances"]:
                temp.append(instance["url"])

            class_output.append(temp)
            output.append(class_output)
            x+=1
        return output
    
def generate(n_classes=20):
    data = {}
    videos = getVideos('Data\\WLASL_v0.3.json', n_classes=n_classes)
    for class_ in videos:
        results = process(class_[1])
        data[class_[0]] = results
        

    return data


# Pads the input sequences with frames < max_frame so that all sequences are of equal length
def pad_sequence(video, max_frame, pad_value):
    # video shape: array of shape (num_frames, 21, 4)

    
    n_frames = video.shape[0]
    if n_frames < max_frame:
        padding = np.full((max_frame-n_frames, 21, 4), pad_value)
        video = np.vstack((video, padding))
    return video


def format_data(data:dict) -> list:
    # This is an overarching function that processes data to the format we need
    # all in one shot

    final_data = []

    for class_ in data:
        class_data = []
        # sequence_lengths = [len(v) for v in data[class_] if len(v)>0]
        for video in data[class_]: # Iterates through each video in each class
            if len(video) == 0:
                continue
            padded_video = pad_sequence(np.array(video), 257, 257)

            # Flatten the pose data for each frame so that we can pass it sequentially into the GRU
            padded_video = np.array([frame.flatten() for frame in padded_video])
            class_data.append(padded_video)
        
        final_data.append(np.array(class_data))

    
    
    return final_data


def trainModel(model: model.GRUModel, epochs:int=5, validate=False):
    pass


