import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Knn_classifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if metric == 'l2':
            distances = np.sqrt(np.sum(np.square(self.train_images - test_image), axis=1))
        elif metric == 'l1':
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        else:
            raise ValueError("Unsupported metric.")

        nearest_neighbor_indices = np.argsort(distances)[:num_neighbors]
        nearest_neighbor_labels = self.train_labels[nearest_neighbor_indices]
        most_common_label = Counter(nearest_neighbor_labels).most_common(1)[0][0]
        return most_common_label


    def classify_images(self, test_images, num_neighbors=3, metric='l2', batch_size=100):
        num_test_samples = len(test_images)
        predicted_labels = []

        for i in range(0, num_test_samples, batch_size):
            batch_test_images = test_images[i:i + batch_size]
            distances = np.zeros((len(batch_test_images), len(self.train_images)))

            for j in range(len(batch_test_images)):
                if metric == 'l2':
                    distances[j] = np.sqrt(np.sum(np.square(self.train_images - batch_test_images[j]), axis=1))
                elif metric == 'l1':
                    distances[j] = np.sum(np.abs(self.train_images - batch_test_images[j]), axis=1)
                else:
                    raise ValueError("Unsupported metric.")

            nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :num_neighbors]
            nearest_neighbor_labels = self.train_labels[nearest_neighbor_indices]

            for labels in nearest_neighbor_labels:
                most_common_label = Counter(labels).most_common(1)[0][0]
                predicted_labels.append(most_common_label)

        return np.array(predicted_labels)

    def accuracy_score(self, predicted, ground_truth):
        correct_predictions = np.sum(predicted == ground_truth)
        return correct_predictions / len(ground_truth)

def load_data(filename, normalize=False):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            label = int(row[0])
            image_data = np.array([float(pixel) for pixel in row[1:2305]])  # Convert to float
            if normalize:
                image_data = image_data / 255.0
            data.append((label, image_data))
    return data

def main():
    # Read and preprocess the data
    train_data = load_data('trainYX.csv', normalize=True)
    test_data = load_data('testYX.csv', normalize=True)

    # Extracting labels and data for convenience
    train_labels, train_images = zip(*train_data)
    test_labels, test_images = zip(*test_data)

    # Initialize the Knn_classifier with training data
    classifier = Knn_classifier(np.array(train_images), np.array(train_labels))

    # Define K values to test (up to 21)
    k_values = list(range(1, 22))
    accuracies = []

    for k in k_values:
        # Classify all test images using the classifier
        predicted_labels = classifier.classify_images(np.array(test_images), num_neighbors=k)

        # Calculate and store the accuracy for the current k
        accuracy = classifier.accuracy_score(predicted_labels, np.array(test_labels))
        accuracies.append(accuracy)

        # Print out the accuracy for the current value of k
        print(f'Accuracy for K={k}: {accuracy:.4f}')

    # Plot the accuracy for each K value
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.xticks(k_values)
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN Accuracy for Different K Values')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
