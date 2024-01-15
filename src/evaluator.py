import numpy as np
import matplotlib.pyplot as plt
import random


class ModelEvaluator:
    @staticmethod
    def intersection_over_union(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        dice = (2.0 * intersection) / union if union != 0 else 0
        return dice

    @staticmethod
    def find_best_threshold(predictions, y_val, step=0.001):
        best_threshold = None
        highest_iou = 0

        for threshold in np.arange(0, 1, step):
            binary_predictions = (predictions > threshold).astype(np.uint8)
            iou_score = ModelEvaluator.intersection_over_union(
                y_val, binary_predictions
            )

            if iou_score > highest_iou:
                highest_iou = iou_score
                best_threshold = threshold

        return best_threshold

    @staticmethod
    def display_results(X_val, y_val, predictions, sample_count=5):
        random.seed(42)
        sample_indexes = random.sample(range(len(X_val)), sample_count)

        plt.figure(figsize=(15, 15))
        for i, idx in enumerate(sample_indexes, 1):
            plt.subplot(sample_count, 3, 3 * i - 2)
            plt.imshow(X_val[idx])
            plt.title("Sample {}: Original Image".format(i))
            plt.subplot(sample_count, 3, 3 * i - 1)
            plt.imshow(y_val[idx], cmap="gray")
            plt.title("Sample {}: Ground Truth Mask".format(i))
            plt.subplot(sample_count, 3, 3 * i)
            plt.imshow(predictions[idx].squeeze(), cmap="gray")
            plt.title("Sample {}: Predicted Mask".format(i))
            plt.tight_layout()
            plt.show()
