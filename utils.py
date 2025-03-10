import mediapipe as mp
import cv2 as cv
import numpy as np
import json
import torch
import model
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

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


def trainModel(model: model.GRUModel, optim:Adam, crit:CrossEntropyLoss, train_loader, valid_loader,  epochs:int=5, validate=False):
    model.train()

    for i in range(epochs):
        running_loss = 0.
        for videos in tqdm(train_loader, desc=f"Training {i+1}/{epochs} epoch"):
            optim.zero_grad()
            X = videos[0]
            y = videos[1]

            out = model(X)
            y = y.long()
            loss = crit(out[:, -1, :], y)
            loss.backward()
            # Monitorting Gradients
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
            # print(f"Gradient Norm: {total_norm}")
            optim.step()
            running_loss += loss.item() / y.size(0)

        print(f'running loss: {running_loss}; Gradient Norm: {total_norm};epoch: {i+1}')






# def trainModel(model:Classifier, epochs:int, validate=False):
#     model.train()
#     running_loss = 0.
#     for i in range(epochs):
#         for images, labels in tqdm(train_loader, desc= f'Training {i + 1}/{epochs} epoch'):
#             optim.zero_grad()
#             output = model(images)
#             loss = crit(output, labels)
#             loss.backward()
#             optim.step()
#             running_loss += loss.item() / labels.size(0)
#         print(f'running loss: {running_loss} epoch; {i + 1}')

#         if validate:
#             model.eval()
#             running_loss = 0.
#             with torch.no_grad():
#                 for images, labels in tqdm(valid_loader, desc= f'Validation {i + 1}/{epochs} epoch'):
#                     output = model(images)
#                     loss = crit(output, labels)
#                     running_loss += loss.item() / labels.size(0)
#                 print(f'running loss: {running_loss} epoch; {i + 1}')

# def testModel(model:Classifier):
#     model.eval()
#     running_loss = 0
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc=f'Testing:'):
#             output = model(images)
#             loss = crit(output, labels)
#             running_loss += loss.item() / labels.size(0)

#         print(f'Running Loss: {running_loss} \n Average Loss Per Batch: {running_loss/test_loader.__len__()}')



