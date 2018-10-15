import os
import sys
import zipfile

import data
import numpy as np
import pylab
import cv2
from PIL import Image
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Input, Flatten, Dense, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from shutil import make_archive, copy2, rmtree


curr_dir = os.getcwd()


def convert_localization_to_annotation(localization_array, row_size=data.IMAGE_ROW_SIZE,
                                       column_size=data.IMAGE_COLUMN_SIZE):
    annotation_list = []
    for localization in localization_array:
        localization = localization[0]

        mask_along_row = np.max(localization, axis=1) > 0.5
        row_start_index = np.argmax(mask_along_row)
        row_end_index = len(mask_along_row) - np.argmax(np.flipud(mask_along_row)) - 1

        mask_along_column = np.max(localization, axis=0) > 0.5
        column_start_index = np.argmax(mask_along_column)
        column_end_index = len(mask_along_column) - np.argmax(np.flipud(mask_along_column)) - 1

        annotation = (
            row_start_index / row_size, (row_end_index - row_start_index) / row_size, column_start_index / column_size,
            (column_end_index - column_start_index) / column_size)
        annotation_list.append(annotation)

    return np.array(annotation_list).astype(np.float32)


def init_model(optimizer, target_num=4, FC_block_num=2, FC_feature_dim=512, dropout_ratio=0.5, learning_rate=0.0001):
    img_rows, img_cols = data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE

    if K.image_data_format() == 'channels_first':
        # input_crop = input_crop.reshape(input_crop.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        # input_crop = input_crop.reshape(input_crop.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    # print('input_shape', input_shape)

    # Get the input tensor
    input_tensor = Input(shape=input_shape)
    print('input_tensor', input_tensor)

    # Convolutional blocks
    pretrained_model = VGG16(include_top=False, weights="imagenet")
    for layer in pretrained_model.layers:
        layer.trainable = False
    output_tensor = pretrained_model(input_tensor)
    print('output_tensor', output_tensor)

    # FullyConnected blocks
    output_tensor = Flatten()(output_tensor)
    for _ in range(FC_block_num):
        output_tensor = Dense(FC_feature_dim, activation="relu")(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Dropout(dropout_ratio)(output_tensor)
    output_tensor = Dense(target_num, activation="sigmoid")(output_tensor)
    print('output_tensor', output_tensor)

    # Define and compile the model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=optimizer, loss="mse")
    # plot_model(model, to_file=os.path.join(d.OPTIMAL_WEIGHTS_FOLDER_PATH, "model.png"), show_shapes=True,
    #            show_layer_names=True)

    return model


def load_dataset(folder_path_list, color_mode_list, batch_size, classes=None, class_mode=None, shuffle=True, seed=None,
                 apply_conversion=False):
    # Get the generator of the dataset
    print('load dataset')
    data_generator_list = []
    for folder_path, color_mode in zip(folder_path_list, color_mode_list):
        data_generator_object = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.2,
            horizontal_flip=True,
            rescale=1.0 / 255)
        data_generator = data_generator_object.flow_from_directory(
            directory=folder_path,
            target_size=(data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE),
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)
        data_generator_list.append(data_generator)

    # Sanity check
    filenames_list = [data_generator.filenames for data_generator in data_generator_list]
    assert all(filenames == filenames_list[0] for filenames in filenames_list)
    # np.save(os.path.join(d.OUTPUT_FOLDER_PATH, 'files.npy'), filenames_list)
    # print(os.path.join(d.OUTPUT_FOLDER_PATH, 'files.npy'))

    if apply_conversion:
        assert len(data_generator_list) == 2
        for X_array, Y_array in zip(*data_generator_list):
            # print(X_array.shape, Y_array.shape)
            yield (X_array, convert_localization_to_annotation(Y_array))
    else:
        for array_tuple in zip(*data_generator_list):
            yield array_tuple


class InspectPrediction(Callback):
    def __init__(self, data_generator_list):
        super(InspectPrediction, self).__init__()

        self.data_generator_list = data_generator_list

    def on_epoch_end(self, epoch, logs=None):
        print('inspect prediction')
        for data_generator_index, data_generator in enumerate(self.data_generator_list, start=1):
            X_array, GT_Y_array = next(data_generator)
            P_Y_array = data.convert_annotation_to_localization(self.model.predict_on_batch(X_array))

            for sample_index, (X, GT_Y, P_Y) in enumerate(zip(X_array, GT_Y_array, P_Y_array), start=1):
                print('X.shape', X.shape)
                print('np.roll', np.rollaxis(X, 0, 3).shape)
                pylab.figure()
                pylab.subplot(1, 3, 1)
                pylab.imshow(X)
                pylab.title("X")
                pylab.axis("off")
                pylab.subplot(1, 3, 2)
                pylab.imshow(GT_Y[0], cmap="gray")
                pylab.title("GT_Y")
                pylab.axis("off")
                pylab.subplot(1, 3, 3)
                pylab.imshow(P_Y[0], cmap="gray")
                pylab.title("P_Y")
                pylab.axis("off")
                pylab.savefig(os.path.join(data.VISUALIZATION_FOLDER_PATH,
                                           "Epoch_{}_Split_{}_Sample_{}.png".format(epoch + 1, data_generator_index,
                                                                                    sample_index)))
                pylab.close()


class InspectLoss(Callback):
    def __init__(self):
        super(InspectLoss, self).__init__()

        self.train_loss_list = []
        self.valid_loss_list = []

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss")
        valid_loss = logs.get("val_loss")
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
        epoch_index_array = np.arange(len(self.train_loss_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_loss_list, "yellowgreen", label="train_loss")
        pylab.plot(epoch_index_array, self.valid_loss_list, "lightskyblue", label="valid_loss")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(data.OUTPUT_FOLDER_PATH, "Loss Curve.png"))
        pylab.close()
        print('inspect loss', 'train_loss', train_loss, 'valid_loss', valid_loss)


def convert_annotation_to_localization(annotation_array, row_size=data.IMAGE_ROW_SIZE,
                                       column_size=data.IMAGE_COLUMN_SIZE):
    localization_list = []
    for annotation in annotation_array:
        localization = np.zeros((row_size, column_size))

        row_start_index = np.max((0, int(annotation[0] * row_size)))
        row_end_index = np.min((row_start_index + int(annotation[1] * row_size), row_size - 1))

        column_start_index = np.max((0, int(annotation[2] * column_size)))
        column_end_index = np.min((column_start_index + int(annotation[3] * column_size), column_size - 1))

        localization[row_start_index:row_end_index + 1, column_start_index:column_end_index + 1] = 1
        localization_list.append(np.expand_dims(localization, axis=0))

    return np.array(localization_list).astype(np.float32)


def unzip_files(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()


def get_train_sample_num(path_to_train_dir):
    train_sample_num = 0
    for fish in os.listdir(path_to_train_dir):
        fish_path = os.path.join(path_to_train_dir, fish)
        train_sample_num += len(os.listdir(fish_path))
    return train_sample_num


def get_optimizer(optimizer, lr):
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr)
    elif optimizer == 'ADAM':
        opt = optimizers.Adam(lr)
    elif optimizer == 'RMSProp':
        opt = optimizers.RMSprop(lr)
    else:
        opt = optimizers.RMSprop(lr)
    return opt


def gen_masked_data(train_data_dir, mask_data_dir, masked_image_data_dir):
    if os.path.exists(masked_image_data_dir):
        return
    os.makedirs(masked_image_data_dir)
    print('masked data dir', masked_image_data_dir)

    dirs = os.listdir(train_data_dir)
    for fish in dirs:
        train_path = os.listdir(os.path.join(train_data_dir, fish))
        for image in train_path:
            # print("item", image)
            if os.path.isfile(os.path.join(train_data_dir, fish, image)):
                real = os.path.join(train_data_dir, fish, image)
                mask = os.path.join(mask_data_dir, fish, image)
                if not os.path.exists(mask):
                    continue
                masked_path = os.path.join(masked_image_data_dir, fish)
                os.makedirs(masked_path, exist_ok=True)
                masked_path = os.path.join(masked_path, image)
                real = cv2.imread(real)
                mask = cv2.imread(mask)
                masked_image = cv2.bitwise_and(real, mask)
                cv2.resize(masked_image, (data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE))
                cv2.imwrite(masked_path, masked_image)
    print('masked data generated')


def run():
    if len(sys.argv) != 3:
        raise ValueError(
            'expected 2 arguments.' + '\neg. python.exe train.py .\Data\Train\Best_hyperparameter_80_percent\data.zip .\hyperparameter.txt')
    data_file_path = sys.argv[1]
    data_folder_path = os.path.dirname(data_file_path)

    unzip_files(data_file_path, data_folder_path)
    train_suffix = data_folder_path.split(sep='\\')[-1]

    # check if necessary folders were extracted
    actual_train_orig_folder_path = os.path.join(data_folder_path, train_suffix, 'train_original')
    actual_train_loc_folder_path = os.path.join(data_folder_path, train_suffix, 'train_localization')
    if not os.path.exists(actual_train_orig_folder_path):
        raise ValueError(actual_train_orig_folder_path + ' does not exist. data.py did not do its job properly')
    if not os.path.exists(actual_train_loc_folder_path):
        raise ValueError(actual_train_loc_folder_path + ' does not exist. data.py did not do its job properly')

    hyp_file_path = sys.argv[2]
    hyp_file = open(hyp_file_path, 'r')
    lines = []
    for line in hyp_file:
        lines.append(line)
        print('line', line)
    model1_params = lines[0].split()
    model2_params = lines[1].split()
    model1_opt = get_optimizer(model1_params[0], float(model1_params[1]))
    model2_opt = get_optimizer(model2_params[0], float(model2_params[1]))

    print("\nInitializing model ...")
    model = init_model(model1_opt)

    train_sample_num = get_train_sample_num(actual_train_orig_folder_path)
    print('\n#train images', train_sample_num)

    model1_file_path = os.path.join(curr_dir, 'model1.h5')

    if os.path.exists(model1_file_path):
        print('model1.h5 exists')
    else:
        print("Performing the training procedure ...")
        train_generator = load_dataset(
            folder_path_list=[actual_train_orig_folder_path, actual_train_loc_folder_path],
            color_mode_list=["rgb", "grayscale"], batch_size=data.BATCH_SIZE, seed=0, apply_conversion=True)
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=data.PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(curr_dir,
                                                   monitor="val_loss",
                                                   save_best_only=True,
                                                   # save_weights_only=True
                                                   )
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=train_sample_num,
                            # validation_data=valid_generator,
                            # validation_steps=valid_sample_num,
                            callbacks=[earlystopping_callback,
                                       modelcheckpoint_callback,
                                       # inspectprediction_callback,
                                       # inspectloss_callback
                                       ],
                            epochs=1,
                            verbose=1)
        model.save(model1_file_path, True)
        print('\nmodel1.h5 generated')


    model2_file_path = os.path.join(curr_dir, 'model2.h5')
    batch_size = data.BATCH_SIZE

    if os.path.exists(model2_file_path):
        print('\nmodel2.h5 exists')
    else:
        masked_dir = os.path.join(data_folder_path, 'masked_data')
        # gen_masked_data(actual_train_orig_folder_path, actual_train_loc_folder_path, masked_dir)
        print('\nLoading InceptionV3 Weights ...')
        InceptionV3_notop = InceptionV3(include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        input_shape=(data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE, 3)
                                        )
        output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
        output = AveragePooling2D((2, 2), strides=(2, 2), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(8, activation='softmax', name='predictions')(output)

        InceptionV3_model = Model(InceptionV3_notop.input, output)

        InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=model2_opt, metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        train_generator = train_datagen.flow_from_directory(
            actual_train_orig_folder_path,
            target_size=(data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE),
            batch_size=batch_size,
            shuffle=True,
            classes=data.FishNames,
            class_mode='categorical'
        )
        InceptionV3_model.fit_generator(
            train_generator,
            steps_per_epoch=train_sample_num,
            epochs=1,
            verbose=1
        )
        InceptionV3_model.save(model2_file_path, True)
        print('model2.h5 generated')
        if os.path.exists(masked_dir):
            rmtree(masked_dir)
    if os.path.exists(actual_train_orig_folder_path):
        rmtree(actual_train_orig_folder_path)

    if os.path.exists(actual_train_loc_folder_path):
        rmtree(actual_train_loc_folder_path)

    if os.path.exists(os.path.join(data_folder_path,train_suffix)):
        rmtree(os.path.join(data_folder_path,train_suffix))


if __name__ == "__main__":
    run()
