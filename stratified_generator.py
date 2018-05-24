import numpy as np
import math

from keras.preprocessing import image


def one_hot_array(index, length):

    one_hot = np.zeros(length)
    one_hot[index] = 1

    return one_hot


def make_stratified_generator(class_paths, target_size=(299, 299), individual_batchsize=8, zoom_factor=2., translation_range=0.05):

    zoom_base = math.sqrt(2) * ((1 - translation_range) ** -1)

    generators = []
    for class_path in class_paths:

        image_data_generator = image.ImageDataGenerator(rotation_range=360,
                                                        zoom_range=(zoom_base, zoom_base * zoom_factor),
                                                        width_shift_range=translation_range,
                                                        height_shift_range=translation_range,
                                                        vertical_flip=True)

        generator = image_data_generator.flow_from_directory(class_path,
                                                             batch_size=individual_batchsize,
                                                             target_size=target_size)
        generators += [generator]

    while True:

        images = []
        for generator in generators:
            images += [next(generator)][0]

        cumulative_batch = np.vstack(images)

        cumulative_labels = np.vstack(
                [np.tile(one_hot_array(i, len(class_paths)), (len(imgs), 1)) for i, imgs in enumerate(images)]
                )

        yield cumulative_batch, cumulative_lables
