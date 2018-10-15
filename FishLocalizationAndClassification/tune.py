import os
import os
import sys

import data
import numpy as np
import train
from keras import backend as K
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Input, Flatten, Dense, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
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


def init_model(optimizer, target_num=4, FC_block_num=2, FC_feature_dim=512, dropout_ratio=0.5):
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

    return model


def train_localization_model(train_generator, train_sample_num, opt, dropratio=0.5):
    model = init_model(opt, dropout_ratio=dropratio)
    # print('\nLoading InceptionV3 Weights ...')

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
    return model


def get_val_data_gen_for_localization(folder_path_list, color_mode_list, classes=None, class_mode=None, shuffle=False,
                                      seed=None):
    # Get the generator of the dataset
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
            batch_size=data.BATCH_SIZE,
            shuffle=shuffle,
            seed=seed)
        print(data_generator.image_shape)

        data_generator_list.append(data_generator)
    return data_generator_list


def predict_localization(data_generator_list, model):
    # Sanity check
    filenames_list = [data_generator.filenames for data_generator in data_generator_list]
    assert all(filenames == filenames_list[0] for filenames in filenames_list)
    assert len(data_generator_list) == 2

    i = 0
    for X_array, Y_array in zip(*data_generator_list):
        print(i)
        i += 1
        # G_Y_array = convert_localization_to_annotation(Y_array)
        prediction = model.predict_on_batch(X_array)
        P_Y_array = convert_annotation_to_localization(prediction)
        andand = np.logical_and(Y_array, P_Y_array)
        oror = np.logical_or(Y_array, P_Y_array)
        area_of_overlap = np.count_nonzero(andand == 1)
        area_of_union = np.count_nonzero(oror == 1)
        IoU = area_of_overlap / (area_of_union * 1.0)
        # for sample_index, (X, GT_Y, P_Y) in enumerate(zip(X_array, G_Y_array, P_Y_array), start=1):
        #     print(sample_index)
        return IoU


def train_classification_model(train_generator, train_sample_num, optimizer):
    print('\ntraining InceptionV3 model ...')
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

    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    InceptionV3_model.fit_generator(
        train_generator,
        steps_per_epoch=train_sample_num,
        epochs=1,
        verbose=1
    )
    return InceptionV3_model


def get_val_data_gen_for_classification(test_data_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE),
        batch_size=data.BATCH_SIZE,
        shuffle=False,  # Important !!!
        classes=None,
        class_mode=None
    )
    return test_generator


def predict_classification(test_data_dir, test_generator, InceptionV3_model, nbr_test_samples):
    print('predicting classification from inceptionV3 model...')

    print('Begin to predict for testing data ...')
    predictions = InceptionV3_model.predict_generator(test_generator)

    y_all = []
    for fish in data.FishNames:
        fish_path = os.path.join(test_data_dir, fish)
        if not os.path.exists(fish_path):
            continue
        y_fish = np.tile(fish, len(os.listdir(fish_path)))
        y_all.extend(y_fish)
    y_all = np.array(y_all)
    # One-Hot-Encode the labels.
    y_all = LabelEncoder().fit_transform(y_all)
    y_all = np_utils.to_categorical(y_all, num_classes=len(data.FishNames))

    predictions = np.array(predictions)
    logloss = log_loss(y_all, predictions)
    print('log loss', logloss)
    accuracy = accuracy_score(y_all, predictions.round(), normalize=False)
    print('accuracy', accuracy / nbr_test_samples)
    return logloss


def run():
    if len(sys.argv) != 3:
        raise ValueError(
            'expected 2 arguments.'
            + "\neg. python.exe tune.py .\Data\Train/Under_90_min_tuning\data.zip .\Data\Validation\Validation_10_percent\data.zip")
    data_file_path = sys.argv[1]
    data_folder_path = os.path.dirname(data_file_path)
    train.unzip_files(data_file_path, data_folder_path)
    train_suffix = data_folder_path.split(sep='\\')[-1]

    val_file_path = sys.argv[2]
    val_folder_path = os.path.dirname(val_file_path)
    val_suffix = val_folder_path.split(sep='\\')[-1]

    train.unzip_files(val_file_path, val_folder_path)

    # check if necessary folders were extracted
    actual_train_orig_folder_path = os.path.join(data_folder_path, train_suffix, 'train_original')
    actual_train_loc_folder_path = os.path.join(data_folder_path, train_suffix, 'train_localization')
    if not os.path.exists(actual_train_orig_folder_path):
        raise ValueError(actual_train_orig_folder_path + ' does not exist. data.py did not do its job properly')
    if not os.path.exists(actual_train_loc_folder_path):
        raise ValueError(actual_train_loc_folder_path + ' does not exist. data.py did not do its job properly')

    actual_val_orig_folder_path = os.path.join(val_folder_path, val_suffix, 'valid_original')
    actual_val_loc_folder_path = os.path.join(val_folder_path, val_suffix, 'valid_localization')
    if not os.path.exists(actual_val_orig_folder_path):
        raise ValueError(actual_val_loc_folder_path + ' does not exist. data.py did not do its job properly')
    if not os.path.exists(actual_val_loc_folder_path):
        raise ValueError(actual_val_loc_folder_path + ' does not exist. data.py did not do its job properly')

    # open tune.txt and hyperparam.txt
    tuning_file_path = os.path.join(curr_dir, 'tuning_results.txt')
    hyp_file_path = os.path.join(curr_dir, 'hyperparameter.txt')
    tuning_file = open(tuning_file_path, 'w')
    tuning_file.write('model1\n')
    tuning_file.close()

    # tune model 1
    print('\n\ntuning model 1...')
    train_sample_num = train.get_train_sample_num(actual_train_orig_folder_path)
    print('#train images', train_sample_num)
    train_generator = train.load_dataset(
        folder_path_list=[actual_train_orig_folder_path, actual_train_loc_folder_path],
        color_mode_list=["rgb", "grayscale"], batch_size=data.BATCH_SIZE, seed=0, apply_conversion=True
    )

    val_generator_list = get_val_data_gen_for_localization(
        folder_path_list=[actual_val_orig_folder_path, actual_val_loc_folder_path],
        color_mode_list=["rgb", "grayscale"]
    )

    learning_rates = [0.001, 0.01, 0.1]
    # learning_rates = [0.1]
    optimizers_list = ['RMSProp', 'SGD', 'ADAM']
    # optimizers_list = ['RMSProp']
    dropratios = [0.5, 0.7, 0.8]
    res = []
    for lr in learning_rates:
        for optimizer in optimizers_list:
            if lr == 0.1 and optimizer=='SGD':
                continue
            if optimizer == 'SGD':
                opt = optimizers.SGD(lr)
            elif optimizer == 'ADAM':
                opt = optimizers.Adam(lr)
            elif optimizer == 'RMSProp':
                opt = optimizers.RMSprop(lr)
            else:
                opt = optimizers.RMSprop(lr)
            for dropratio in dropratios:
                print('lr: ', lr, 'optimizer', optimizer, 'dropratio', dropratio)
                print('\nlr: ', lr, 'optimizer', optimizer)
                model_localization = train_localization_model(train_generator, train_sample_num, opt, dropratio)
                IoU = predict_localization(
                    data_generator_list=val_generator_list,
                    model=model_localization
                )
                print('IoU', IoU)
                res.append(IoU)
                tuning_file = open(tuning_file_path, 'a')
                tuning_file.write(repr(optimizer) + '\t' + repr(lr) +'\t' + repr(dropratio) + '\t' + repr(IoU) + '\n')
                tuning_file.close()

    min_idx = res.index(max(res))
    min_lr_idx = int(min_idx / (len(optimizers_list) * len(dropratios)))
    min_idx_bar = int(min_idx % (len(optimizers_list) * len(dropratios)))
    min_opt_idx = int(min_idx_bar / len(optimizers_list))
    min_drop_idx = int(min_idx_bar % len(optimizers_list))

    hyp_file = open(hyp_file_path, 'w')
    hyp_file.write(optimizers_list[min_opt_idx] + '\t' + repr(learning_rates[min_lr_idx]) + '\t' + repr(
        dropratio[min_drop_idx]) + '\n')
    hyp_file.close()
    # return

    print('\n\ntuning model 2')
    tuning_file = open(tuning_file_path, 'a')
    tuning_file.write('model2\n')
    tuning_file.close()
    val_sample_num = train.get_train_sample_num(actual_val_orig_folder_path)
    print('\n#val images', val_sample_num)
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
        batch_size=data.BATCH_SIZE,
        shuffle=True,
        classes=data.FishNames,
        class_mode='categorical'
    )
    val_generator = get_val_data_gen_for_classification(actual_val_orig_folder_path)
    learning_rates = [0.1]
    # learning_rates = [0.001]
    optimizers_list = ['RMSProp', 'SGD', 'ADAM']
    # optimizers_list = ['RMSProp', 'ADAM']

    res = []
    for lr in learning_rates:
        for optimizer in optimizers_list:
            if lr == 0.1 and optimizer=='SGD':
                continue
            if optimizer == 'SGD':
                opt = optimizers.SGD(lr)
            elif optimizer == 'ADAM':
                opt = optimizers.Adam(lr)
            elif optimizer == 'RMSProp':
                opt = optimizers.RMSprop(lr)
            else:
                opt = optimizers.RMSprop(lr)
            print('lr: ', lr, 'optimizer', optimizer)
            model_classification = train_classification_model(train_generator, train_sample_num, opt)
            logloss = predict_classification(actual_val_orig_folder_path, val_generator, model_classification,
                                             val_sample_num)
            print('log loss', logloss)
            res.append(logloss)
            tuning_file = open(tuning_file_path, 'a')
            tuning_file.write(
                repr(optimizer) + '\t' + repr(lr) +  '\t' + repr(logloss) + '\n')
            tuning_file.close()

    min_idx = res.index(min(res))
    min_lr_idx = int(min_idx / len(optimizers_list))
    min_lr_idx = int(min_idx % len(optimizers_list))

    hyp_file = open(hyp_file_path, 'a')
    hyp_file.write(optimizers_list[min_opt_idx] + '\t' + repr(learning_rates[min_lr_idx]) + '\n')
    hyp_file.close()

    if os.path.exists(actual_train_orig_folder_path):
        rmtree(actual_train_orig_folder_path)
    if os.path.exists(actual_train_loc_folder_path):
        rmtree(actual_train_loc_folder_path)
    if os.path.exists(actual_val_orig_folder_path):
        rmtree(actual_val_orig_folder_path)
    if os.path.exists(actual_val_loc_folder_path):
        rmtree(actual_val_loc_folder_path)
    if os.path.exists(os.path.join(data_folder_path, train_suffix)):
        rmtree(os.path.join(data_folder_path, train_suffix))
    if os.path.exists(os.path.join(data_folder_path, val_suffix)):
        rmtree(os.path.join(data_folder_path, val_suffix))


if __name__ == "__main__":
    run()
