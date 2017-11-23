import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import numpy as np
from skimage import io, data, transform

def load_data(data_dir):
    """

    :param data_dir: the path of train_data
    :return: images: images data read by skimage
             labels_: labels for each picture

    """
    data = open(data_dir)
    lines = data.readlines()
    images_paths = []
    labels = []
    for line in lines:
        path_label = line[1:]
        path, label = path_label.split(' ')
        images_paths.append('image' + path)
        labels.append(float(label[0]))

    images = []
    for i in images_paths:
        images.append(skimage.data.imread(i))

    return images, labels

# have a glance through the first 100 images from a specific class
def display_images_and_labels(images, labels, label_value):
    """

    :param images: list, paths for images
    :param labels: list, labels for images
    :param label_value: str:{0,1,2,3,4} for 5 classes
    :return: show a 10 * 10 plt containing 100 image from a specific class
    """
    count = 1
    plt.figure(figsize=(50, 50))
    for i in range(len(labels)):
        if count > 100:
            break
        if labels[i] + '' == label_value:
            image = images[i]
            plt.subplot(10, 10, count)
            plt.axis('off')
            count += 1
            plt.imshow(image)
    plt.show()