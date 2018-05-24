import numpy as np
import time
import tqdm
import json
import logging
import argparse

from pathlib import Path

from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger

from stratified_generator import stratified_generator


def write_status(filename, epoch_nr, training_error, validation_error, training_loss, validation_loss):

    info = {'epoch_nr': epoch_nr,
            'timestamp': time.time(),
            'training_error': training_error,
            'validation_error': validation_error,
            'training_loss': training_loss,
            'validation_loss': validation_loss}

    with open(filename, 'w') as fp:
        json.dump(info, fp)


def make_generators(path_images, positive_class='merger', negative_class='noninteracting'):

    training_gen = make_stratified_generator(Path(path_images) / positive_class / 'training',
                                             Path(path_images) / negative_class / 'training')

    validation_gen = make_stratified_generator(Path(path_images) / positive_class / 'validation',
                                               Path(path_images) / negative_class / 'validation')

    return training_gen, validation_gen


def training_loop(model, nr_epochs, path_images, training_type, steps_per_epoch = (6000 * 2/4), validation_steps = (6000 * 1/4)):

    training_gen, validation_gen=make_generators(path_images)

    for epoch in tqdm.tqdm(range(nr_epochs), total = nr_epochs):

        logging.info("starting {training_type} epoch #{epoch}".format(training_type, epoch))
        history=model.fit_generator(training_gen, steps_per_epoch = steps_per_epoch, epochs = 1,
                                    validation_data = validation_gen, validation_steps = validation_steps)
        logging.info("finished {training_type} epoch #{epoch}".format(training_type, epoch))

        model.save(Path(path_checkpoints) / '{training_type}_{epoch}.checkpoint'.format(training_type, epoch))

        training_error, validation_error = (1. - history.history['acc'][0]), (1. - history.history['val_acc'][0])
        training_loss, validation_loss = history.history['loss'][0], history.history['val_loss'][0]

        write_status(Path(path_statuses) / '{training_type}_{epoch}.status'.format(training_type, epoch),
                     epoch,
                     training_error,
                     validation_error,
                     training_loss,
                     validation_loss)

        return model


def preparation_training(path_images, path_checkpoints, path_statuses, nr_epochs = 40):

    if mode == 'transferlearning':
        xception=Xception(weights = 'imagenet', include_top = False)
    else:
        xception=Xception(weights = None, include_top = False)

    output=GlobalAveragePooling2D()(xception.output)
    output=Dense(1024, activation = 'relu')(output)
    output=Dense(2, activation = 'softmax')(output)
    model=Model(input = xception.input, output = output)

    for layer in xception.layers:
        layer.trainable=False  # freeze all layers except the new FC layers at the end

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics=['accuracy'])

    model = training_loop(model, nr_epochs, path_images, training_type='preparation')

    model.save(Path(path_checkpoints) / 'main_0.checkpoint')


def main_training(path_images, path_checkpoints, path_statuses, current_epoch, nr_epochs=100000):

    current_checkpoint_filename = Path(path_checkpoints) / 'main_{current_epoch}.checkpoint'.format(current_epoch)
    model = load_model(current_checkpoint_filename)

    for layer in model.layers:
       layer.trainable = True # unfreeze all layers

    model.compile(optimizer=SGD(lr=0.5 * 3e-5, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    training_loop(model, nr_epochs, path_images, training_type='main')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_images', required=True)
    parser.add_argument('--path_checkpoints', required=True)
    parser.add_argument('--path_statuses', required=True)
    parser.add_argument('--mode', required=True)
    args = parser.parse_args()

    path_images, path_checkpoints, path_statuses, mode = (args.path_images,
                                                          args.path_checkpoints,
                                                          args.path_statuses,
                                                          args.mode)

    logging.basicConfig(filename='LOG.log', level=logging.DEBUG)


    logging.info(("started with parameters",
                 "path_images: '{path_images}''".format(path_images),
                 "path_checkpoints: '{path_checkpoints}''".format(path_checkpoints),
                 "mode: '{mode}''".format(mode)))


    logging.info("starting new training")
    preparation_training(path_images, path_checkpoints, path_statuses, mode)

    logging.info("starting main training")
    main_training(path_images, path_checkpoints, path_statuses, 0)
