import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# Function to draw landmarks on the image
def draw_landmarks(image, landmarks, landmark_names):
    h, w = image.shape[:2]
    landmarks = landmarks * [w, h]

    for (x, y), name in zip(landmarks, landmark_names):
        if x >= 0 and y >= 0:  # Assuming that negative values mean 'not visible'
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(image, name, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    image_dir = 'dataset/'
    landmark_file = 'annotations/list_landmarks.txt'
    model_path = 'models/my_clothing_landmark_model.h5'
    test_image_path = 'dataset/img_00000001.jpg'




if __name__ == "__main__":
    main()
