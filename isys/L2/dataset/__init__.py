#! /usr/bin/python3

import os
import h5py

__dataset_file_path = os.path.join(os.path.dirname(__file__), 'build/dataset.hdf5')

def __load():
    with h5py.File(__dataset_file_path, 'r') as data_file:
        data_images = data_file['images'][...]
        dataset_labels = data_file['labels'][...]
        return data_images, dataset_labels

def __split(images, labels, test_split_ratio=None):
    test_split_ratio = test_split_ratio if test_split_ratio is not None else 1/6  # fallback
    number = int((1-test_split_ratio)*images.shape[0])
    x_train, x_test = images[:number], images[number:]
    y_train, y_test = labels[:number], labels[number:]
    
    return (x_train, y_train), (x_test, y_test)

def load_data(test_split_ratio=1/5):
    images, labels = __load()
    data = __split(images, labels, test_split_ratio)
    return data

    
