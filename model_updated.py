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
import cv2
import os
import re
import yt_dlp


def remove_duplicates(json_file):
# this function accepts a json file as a parameter to check for, and eliminate, and duplicated
# for our dataset, we are eliminating duplicated based on the 'org_text' value
    with open(json_file, 'r') as file:
        data = json.load(file)

    seen_texts = set()
    unique_data = []

    for row in data:
        org_text = row.get('org_text')
        if org_text not in seen_texts:
            seen_texts.add(org_text)
            unique_data.append(row)

    with open('filtered_' + json_file, 'w') as outfile:
        json.dump(unique_data, outfile, indent=4)

    print(f"Filtered data with duplicates removed has been saved to 'filtered_{json_file}'.")


def video_to_images(url, output_folder, name, start_time=None, end_time=None):
# this function accepts information from the asl dataset as parameters including the url, name, and start/end times
# the YouTube videos are downloaded using the corresponding url, and will return an error if there is a url issue
# every 10th frame from the video (only within the specified start and end times) is then saved
    try:
        print(f"Attempting to download video from URL: {url}")
        ydl_opts = {
            'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
            'format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        video_file = next((f for f in os.listdir(output_folder) if f.endswith('.mp4')), None)
        if video_file is None:
            print(f"Error: No video file found in {output_folder}")
            return False

        video_path = os.path.join(output_folder, video_file)
        print(f"Downloaded video to: {video_path}")
    except Exception as e:
        print(f"Failed to download video: {url}. Error: {e}")
        return False

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if start_time is not None:
        start_frame = int(start_time * fps)
    else:
        start_frame = 0

    if end_time is not None:
        end_frame = int(end_time * fps)
    else:
        end_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count < start_frame:
            frame_count += 1
            continue

        if frame_count > end_frame:
            break

        if frame_count % 10 == 0:
            filename = f"frame_{frame_count}_{name}.jpg"
            frame_path = os.path.join(output_folder, filename)  # Save to the correct subfolder
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame to: {frame_path}")

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()
    os.remove(video_path)

    return True


def extract_features(image):
# this function accepts an image (attained from the YouTube videos) as a parameter and is called in load_data()
# it uses this image and the HOG technique for object detection and feature extraction
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features


def load_data(base_folder):
# this function accepts a base folder (in our case the 'Videos-to-Images Ouput' folder) as a parameter
# it loops through this folder and its subfolders to load each asl image (each subfolder is one word/phrase)
    data = []
    labels = []

    for sign_folder in os.listdir(base_folder):
        sign_folder_path = os.path.join(base_folder, sign_folder)
        if os.path.isdir(sign_folder_path):
            label = sign_folder

            for img_file in os.listdir(sign_folder_path):
                img_path = os.path.join(sign_folder_path, img_file)

                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img = cv2.imread(img_path)

                    if img is not None:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        img_resized = cv2.resize(img_gray, (128, 128))

                        features = extract_features(img_resized)

                        data.append(features)
                        labels.append(label)
                    else:
                        print(f"Warning: Could not read image {img_path}")

    return np.array(data), np.array(labels)


def clean_folder_names(name):
# this function accepts a subfolder name ('org_text in our dataset) as a parameter
# it then removes invalid special characters from the subfolder names
    name = name.replace('\n', '').replace('\r', '')
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def main():
    # remove repeated words/phrases from dataset
    remove_duplicates('MSASL_val.json')
    # open new data file (without duplicates)
    with open('filtered_MSASL_val.json', 'r') as json_file:
        video_info = json.load(json_file)
        # set name for the folder that will hold the images attained for each sign
        output_folder = 'Video-to-Image Outputs'
        video_num = 1

        # loop through rows in asl dataset
        for row in video_info:
            # set important features to their corresponding value in the json file
            url = row.get('url')
            name = row.get('org_text')
            start_time = row.get('start_time')
            end_time = row.get('end_time')
            # remove additional special characters from subfolder names
            subfolder_name = clean_folder_names(name)
            # set path for each word/phrase subfolder
            subfolder_path = os.path.join(output_folder, subfolder_name)

            # check if url exists (some videos are unavailable/private)
            if url:
                # download videos and use the video_to_images() function to save the created images
                success = video_to_images(url, subfolder_path, video_num, start_time, end_time)
                # If download is successful, create subfolder and process video
                if success:
                    # save images to corresponding subfolder
                    os.makedirs(subfolder_path, exist_ok=True)
                    # print message to console to keep track of progress
                    print(f"Video number {video_num} is complete.")
                    video_num += 1

    # set base folder to location of asl images and load data
    base_folder = 'Video-to-Image Outputs'
    X, y = load_data(base_folder)
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train the SVM model
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    # test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print accuracy score
    print(f"Model accuracy: {accuracy * 100:.2f}%")


main()






