# Emotion-Detection-Using-K-Nearest-Neighbors-KNN-

s the implementation and evaluation of a K-Nearest Neighbors
(KNN) classifier for the task of emotion detection from facial images. Each 48x48 pixel image in our
dataset represents one of seven emotions, encoded as integers 0 through 6. The model's
performance was assessed across various values of K to determine the optimal number of neighbors
for classification.
Methodology: The KNN classifier was implemented from scratch in Python, using the L2 (Euclidean)
distance metric to compare pixel values between images. The dataset comprised two CSV files:
'trainYX.csv' for training and 'testYX.csv' for testing. Prior to classification, pixel values were
normalized to the [0, 1] range to ensure uniformity in distance calculations.


