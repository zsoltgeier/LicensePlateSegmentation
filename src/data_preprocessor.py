import numpy as np
import cv2
import os
import glob
from lxml import etree
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, image_dir, annotation_dir, image_size=128):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size

    def load_images(self):
        data_path = os.path.join(self.image_dir, "*g")
        files = sorted(glob.glob(data_path))
        images = [self._process_image(cv2.imread(f)) for f in files]
        return np.array(images, dtype=int)

    def load_annotations(self):
        files = sorted(glob.glob(os.path.join(self.annotation_dir, "*.xml")))
        annotations = [self._resize_annotation(f) for f in files]
        return np.array(annotations, dtype=int)

    def _process_image(self, img):
        return cv2.resize(img, (self.image_size, self.image_size))

    def _resize_annotation(self, file_path):
        tree = etree.parse(file_path)
        for dim in tree.xpath("size"):
            width = int(dim.xpath("width")[0].text)
            height = int(dim.xpath("height")[0].text)

        for dim in tree.xpath("object/bndbox"):
            xmin = int(dim.xpath("xmin")[0].text) / (width / self.image_size)
            ymin = int(dim.xpath("ymin")[0].text) / (height / self.image_size)
            xmax = int(dim.xpath("xmax")[0].text) / (width / self.image_size)
            ymax = int(dim.xpath("ymax")[0].text) / (height / self.image_size)

        return [int(xmax), int(ymax), int(xmin), int(ymin)]

    def create_binary_masks(self, annotations):
        masks = []
        for annotation in annotations:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            for bbox in annotation:
                xmax, ymax, xmin, ymin = bbox
                mask[ymin:ymax, xmin:xmax] = 1
            masks.append(mask)
        return np.array(masks)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
