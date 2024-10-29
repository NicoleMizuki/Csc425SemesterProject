import os
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import cv2
import json
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from pytube import YouTube

from pytube import YouTube
import cv2
import os


# this function has been altered slightly to try and fix the youtube link issue
def video_to_images(url, output_folder, name):
    # Download the video from YouTube
    try:
        print(f"Attempting to download video from URL: {url}")
        yt = YouTube(url)
        video = yt.streams.filter(file_extension='mp4').first()

        # Check if video is available
        if video is None:
            print(f"Error: Video is private or unavailable for URL: {url}")
            return

        video_path = video.download(output_folder)  # Downloads video to output_folder
        print(f"Downloaded video to: {video_path}")
    except Exception as e:
        # Print specific error messages for debugging
        if "private video" in str(e):
            print(f"Error downloading video: {url} is a private video")
        elif "unavailable" in str(e):
            print(f"Error downloading video: {url} is unavailable")
        elif "HTTP Error 400" in str(e):
            print(f"Error downloading video: Bad request for URL: {url}")
        else:
            print(f"Error downloading video: {e}")
        return

    # Now process the downloaded video
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % 10 == 0:  # Save every 10th frame
            filename = f"frame_{frame_count}_{name}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()


# moved to main function
# json_file_path = 'MSASL_val.json'
# # TODO
# # output_folder = make output folder
#
# video_num = 1
# with open(json_file_path, 'r') as json_File:
#     load_file = json.load(json_File)
#
#     for video_info in load_file:
#         url = video_info.get('url')
#         name = video_info.get('org_text')
#         subfolder_name = name
#         subfolder_path = os.path.join(output_folder, subfolder_name)
#         os.makedirs(subfolder_path)
#
#         if url:
#             video_to_images(url, subfolder_name, video_num)
#             video_num += 1
#         else:
#             print(f"No URL found for video {video_num}")


def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, multichannel=True)
    return features


def load_data_from_folders(base_folder):
    data = []
    labels = []

    # Loop through each subfolder (representing each sign)
    for sign_folder in os.listdir(base_folder):
        sign_folder_path = os.path.join(base_folder, sign_folder)
        if os.path.isdir(sign_folder_path):
            # Each folder represents a sign (label)
            label = sign_folder

            for img_file in os.listdir(sign_folder_path):
                img_path = os.path.join(sign_folder_path, img_file)

                img = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img_resized = cv2.resize(img_gray, (128, 128))

                features = extract_hog_features(img_resized)

                data.append(features)
                labels.append(label)

    return np.array(data), np.array(labels)


def main():
    video_to_images("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "output_folder", "test_video")
    with open('MSASL_val.json', 'r') as json_file:
        video_info = json.load(json_file)
        output_folder = 'output_folder'
        video_num = 1

        for row in video_info:
            url = row.get('url')
            name = row.get('org_text')
            subfolder_name = name
            subfolder_path = os.path.join(output_folder, subfolder_name)

            # Create the subfolder, do not raise an error if it already exists
            os.makedirs(subfolder_path, exist_ok=True)

            if url:
                video_to_images(url, subfolder_path, video_num)  # Note: Use subfolder_path here
                video_num += 1
            else:
                print(f"No URL found for video {video_num}")

    base_folder = output_folder
    X, y = load_data_from_folders(base_folder)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

main()


