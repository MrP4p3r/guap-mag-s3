#! /usr/bin/python3

import os
import sys

import gzip

import subprocess as sp
from PIL import Image
import numpy as np

import traceback

import h5py


WORK_DIR = os.path.dirname(__file__)
SOURCE_DIR = os.path.join(WORK_DIR, r'./src-png')
BUILD_DIR  = os.path.join(WORK_DIR, r'./build')
dataset_file_path = os.path.join(BUILD_DIR, r'dataset.hdf5')


def main():
    src_join = lambda img_name: os.path.join(SOURCE_DIR, img_name)

    # list of files to split
    files = list(map(src_join, os.listdir(SOURCE_DIR)))
    print(list(enumerate(files)))
    if not files: exit()
    
    # dataset shapes
    num_images_per_file = 5*5
    num_records = len(files) * num_images_per_file
    dataset_shape_images = (num_records, 32, 32)
    dataset_shape_labels = (num_records,)
    
    # try created build directory
    try:
        os.mkdir(BUILD_DIR)
    except:
        pass

    with h5py.File(dataset_file_path, 'w') as data_file:
        dataset_images = data_file.create_dataset('images', dataset_shape_images, dtype='u1')
        dataset_labels = data_file.create_dataset('labels', dataset_shape_labels, dtype='u1')

        # crop and load
        images_array = np.array([], dtype='uint8')
        labels_array = np.repeat(np.arange(len(files), dtype='uint8'), num_images_per_file)
        for file in files:
            p = sp.Popen(['convert', file, '-resize', '160x160', '-crop', '32x32', 'gray:-' ], stdout=sp.PIPE)
            raw_data = p.stdout.read()
            images_array = np.concatenate([images_array, np.fromstring(raw_data, dtype='uint8')])
            
        images_array = images_array.reshape(dataset_shape_images)
        
        shuffle_order = np.arange(labels_array.shape[0])
        np.random.shuffle(shuffle_order)
        
        images_array = images_array[shuffle_order]
        labels_array = labels_array[shuffle_order]

        dataset_images[...] = images_array
        dataset_labels[...] = labels_array
    
    return 0
    

if __name__ == '__main__':
    sys.exit(main())

