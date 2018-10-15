import os
import os
import sys

import data
import numpy as np
import pylab
import train
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from shutil import make_archive, copy2, rmtree

curr_dir = os.getcwd()


def predict_inceptionv3(test_data_dir, inceptionv3_saved_model, nbr_test_samples):
    print('Loading model and weights from training process ...')
    InceptionV3_model = load_model(inceptionv3_saved_model)

    # test data generator for prediction
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE),
        batch_size=data.BATCH_SIZE,
        shuffle=False,  # Important !!!
        classes=None,
        class_mode=None)

    print('Begin to predict for testing data ...')
    predictions = InceptionV3_model.predict_generator(test_generator)
    # np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)

    # print(os.listdir(test_data_dir))
    y_all = []
    for fish in data.FishNames:
        fish_path = os.path.join(test_data_dir, fish)
        if not os.path.exists(fish_path):
            continue
        y_fish = np.tile(fish, len(os.listdir(fish_path)))
        # print('len ', len(y_fish))
        y_all.extend(y_fish)
    y_all = np.array(y_all)
    # print('y_all', y_all)
    # One-Hot-Encode the labels.
    y_all = LabelEncoder().fit_transform(y_all)
    y_all = np_utils.to_categorical(y_all, num_classes=len(data.FishNames))
    # print(y_all)
    # print(predictions)
    predictions = np.array(predictions)
    print('shape', y_all.shape, predictions.shape)
    # print(np.isnan(predictions))
    # print(np.isinf(predictions))
    logloss = log_loss(y_all, predictions)
    print('log loss', logloss)
    accuracy = accuracy_score(y_all, predictions.round(), normalize=False)
    print('accuracy', accuracy / nbr_test_samples)
    return logloss, accuracy


def predict(model, test_data_gen_list):
    print('test prediction')
    for data_generator_index, data_generator in enumerate(test_data_gen_list, start=1):
        # print(data_generator.filename)
        X_array, GT_Y_array = next(data_generator)
        prediction = model.predict_on_batch(X_array)
        print(type(prediction), prediction.shape, X_array.shape, GT_Y_array.shape)
        print()

        # json_file_path = os.path.join(VISUALIZATION_FOLDER_PATH, "test_res_in_json.json")
        # with open(json_file_path, 'w') as fp:
        #     json.dump(prediction, fp)

        P_Y_array = data.convert_annotation_to_localization(prediction)
        print(type(P_Y_array))
        print(P_Y_array.shape)
        idx = np.where(P_Y_array[0] == 1)
        # print(idx)
        print(idx[0][0], idx[1][0], idx[2][0])
        print(idx[0][0], idx[1][-1], idx[2][-1])
        print(idx[1][-1] - idx[1][0], idx[2][-1] - idx[2][0])
        idx = np.where(GT_Y_array[0] == 1)
        # print(idx)
        print(idx[0][0], idx[1][0])
        print(idx[0][0], idx[1][-1])
        print(idx[0][-1] - idx[0][0], idx[1][-1] - idx[1][0])

        for sample_index, (X, GT_Y, P_Y) in enumerate(zip(X_array, GT_Y_array, P_Y_array), start=1):
            print('X.shape', X.shape, 'GT_Y[0].shape', GT_Y[0].shape, 'GT_Y.shape', GT_Y.shape)
            print('P_Y[0].shape', P_Y[0].shape, 'P_Y.shape', P_Y.shape)
            # print('np.roll', np.rollaxis(X, 0, 3).shape)
            pylab.figure()
            pylab.subplot(1, 3, 1)
            pylab.imshow(X)
            pylab.title("X")
            pylab.axis("off")
            pylab.subplot(1, 3, 2)
            pylab.imshow(np.rollaxis(GT_Y, 2, 0)[0], cmap="gray")
            pylab.title("GT_Y")
            pylab.axis("off")
            pylab.subplot(1, 3, 3)
            pylab.imshow(P_Y[0], cmap="gray")
            pylab.title("P_Y")
            pylab.axis("off")
            pylab.savefig(os.path.join(data.VISUALIZATION_FOLDER_PATH,
                                       "Split_{}_Sample_{}.png".format(data_generator_index,
                                                                       sample_index)))
            pylab.close()
            print('type', type(X), type(np.rollaxis(GT_Y, 2, 0)[0]))
            mask = np.rollaxis(GT_Y, 2, 0)[0]
            # np.where(mask[..., None] == 0, X, [0, 0, 255])
            # masked_image = cv2.bitwise_and(X, np.rollaxis(GT_Y,2,0)[0])
            # cv2.imwrite(os.path.join(VISUALIZATION_FOLDER_PATH,
            #                            "mask_{}_Sample_{}.png".format(data_generator_index,
            # sample_index)),X)


def predict_inceptionv3_model(test_data_dir, inceptionv3_saved_model):
    # test data generator for prediction
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(data.IMAGE_ROW_SIZE, data.IMAGE_COLUMN_SIZE),
        batch_size=data.BATCH_SIZE,
        shuffle=False,  # Important !!!
        classes=None,
        class_mode=None)

    test_image_list = test_generator.filenames
    nbr_test_samples = len(test_image_list)
    # print('test image list',len(test_image_list),test_image_list)

    print('Loading model and weights from training process ...')
    InceptionV3_model = load_model(inceptionv3_saved_model)

    print('Begin to predict for testing data ...')
    predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
    # np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)

    print(os.listdir(test_data_dir))
    y_all = []
    for fish in data.FishNames:
        fish_path = os.path.join(test_data_dir, fish)
        if not os.path.exists(fish_path):
            continue
        y_fish = np.tile(fish, len(os.listdir(fish_path)))
        print('len ', len(y_fish))
        y_all.extend(y_fish)
    y_all = np.array(y_all)
    print('y_all', y_all)
    # One-Hot-Encode the labels.
    y_all = LabelEncoder().fit_transform(y_all)
    y_all = np_utils.to_categorical(y_all, num_classes=len(data.FishNames))
    print(y_all)
    # print(predictions)
    predictions = np.array(predictions)
    print('shape', y_all.shape, predictions.shape)
    logloss = log_loss(y_all, predictions)
    print('log loss', logloss)
    accuracy = accuracy_score(y_all, predictions.round(), normalize=False)
    print('accuracy', accuracy)
    return logloss, accuracy


def ajairatest(model, test_data_gen_list):
    filenames = np.load(os.path.join(data.OUTPUT_FOLDER_PATH, 'files.npy'))
    print(filenames)
    test_mask_path = os.path.join(data.OUTPUT_FOLDER_PATH, 'test_mask')
    if not os.path.exists(test_mask_path):
        os.makedirs(test_mask_path)
    print('test prediction')
    i = 0
    for data_generator_index, data_generator in enumerate(test_data_gen_list, start=1):
        X_array, GT_Y_array = next(data_generator)
        prediction = model.predict_on_batch(X_array)
        print(type(prediction), prediction.shape, X_array.shape, GT_Y_array.shape)
        print()

        P_Y_array = train.convert_annotation_to_localization(prediction)
        # print(type(P_Y_array))
        print(P_Y_array.shape)
        idx = np.where(P_Y_array[0] == 1)
        # print(idx)
        # print(idx[0][0], idx[1][0], idx[2][0])
        # print(idx[0][0], idx[1][-1], idx[2][-1])
        # print(idx[1][-1] - idx[1][0], idx[2][-1] - idx[2][0])
        idx = np.where(GT_Y_array[0] == 1)
        # print(idx)
        # print(idx[0][0], idx[1][0])
        # print(idx[0][0], idx[1][-1])
        # print(idx[0][-1] - idx[0][0], idx[1][-1] - idx[1][0])

        for sample_index, (X, GT_Y, P_Y) in enumerate(zip(X_array, GT_Y_array, P_Y_array), start=1):
            print('X.shape', X.shape, 'GT_Y[0].shape', GT_Y[0].shape, 'GT_Y.shape', GT_Y.shape)
            print('P_Y[0].shape', P_Y[0].shape, 'P_Y.shape', P_Y.shape)
            # print('np.roll', np.rollaxis(X, 0, 3).shape)
            pylab.figure()
            pylab.subplot(1, 3, 1)
            pylab.imshow(X)
            pylab.title("X")
            pylab.axis("off")
            pylab.subplot(1, 3, 2)
            pylab.imshow(np.rollaxis(GT_Y, 2, 0)[0], cmap="gray")
            pylab.title("GT_Y")
            pylab.axis("off")
            pylab.subplot(1, 3, 3)
            pylab.imshow(P_Y[0], cmap="gray")
            pylab.title("P_Y")
            pylab.axis("off")
            pylab.savefig(os.path.join(data.VISUALIZATION_FOLDER_PATH,
                                       "Split_{}_Sample_{}.png".format(data_generator_index,
                                                                       sample_index)))
            pylab.close()
            print('type', type(X), type(np.rollaxis(GT_Y, 2, 0)[0]))
            mask = np.rollaxis(GT_Y, 2, 0)[0]
            # mask = np.where(mask[..., None] == 0, X, [0, 0, 255])
            # mask = [mask,mask,mask]
            # mask = np.array(mask)
            # mask = np.rollaxis(mask,0,2)
            # print(mask.shape, X.shape)
            # masked_image = cv2.bitwise_and(X, mask)

            img = Image.fromarray(np.array(mask).astype(np.uint8))
            # img.save('test.jpg')
            # img = Image.fromarray(mask)
            # print()
            tt = filenames[0][i].split("\\")
            folder_name = tt[0]
            image_name = tt[1].split('.')[0]
            print(folder_name, image_name)
            # print(filenames[data_generator_index][i],type(filenames[data_generator_index][i]))
            i += 1
            mask_path = os.path.join(test_mask_path, folder_name)
            print(mask_path, image_name + '.jpg')
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            img.save(os.path.join(mask_path, image_name + '.jpg'))


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


def predict_model1(folder_path_list, color_mode_list, model, classes=None, class_mode=None, shuffle=False, seed=None):
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

    # Sanity check
    filenames_list = [data_generator.filenames for data_generator in data_generator_list]
    assert all(filenames == filenames_list[0] for filenames in filenames_list)
    assert len(data_generator_list) == 2

    i = 0
    for X_array, Y_array in zip(*data_generator_list):
        print(i)
        i += 1
        G_Y_array = convert_localization_to_annotation(Y_array)
        # print(X_array.shape, Y_array.shape, G_Y_array.shape)
        prediction = model.predict_on_batch(X_array)
        # print(prediction.shape)
        P_Y_array = convert_annotation_to_localization(prediction)
        # print(P_Y_array.shape)
        andand = np.logical_and(Y_array, P_Y_array)
        oror = np.logical_or(Y_array, P_Y_array)
        area_of_overlap = np.count_nonzero(andand == 1)
        area_of_union = np.count_nonzero(oror == 1)
        IoU = area_of_overlap / (area_of_union * 1.0)
        # for sample_index, (X, GT_Y, P_Y) in enumerate(zip(X_array, G_Y_array, P_Y_array), start=1):
        #     print(sample_index)
        return IoU


def run():
    if len(sys.argv) != 4:
        raise ValueError(
            'expected 3 arguments.' + '\neg. python.exe test.py .\Data\Test\Test_10_percent\data.zip .\model1.h5 .\model2.h5')
    data_file_path = sys.argv[1]
    data_folder_path = os.path.dirname(data_file_path)
    test_suffix = data_folder_path.split(sep='\\')[-1]

    train.unzip_files(data_file_path, data_folder_path)

    # check if necessary folders were extracted
    actual_test_orig_folder_path = os.path.join(data_folder_path, test_suffix, 'test_original')
    actual_test_loc_folder_path = os.path.join(data_folder_path, test_suffix, 'test_localization')
    if not os.path.exists(actual_test_orig_folder_path):
        raise ValueError(actual_test_orig_folder_path + ' does not exist. data.py did not do its job properly')
    if not os.path.exists(actual_test_loc_folder_path):
        raise ValueError(actual_test_loc_folder_path + ' does not exist. data.py did not do its job properly')

    model1_file = sys.argv[2]
    model2_file = sys.argv[3]
    if not os.path.exists(model1_file):
        raise ValueError(model1_file + ' does not exist. train.py did not do its job properly')
    if not os.path.exists(model2_file):
        raise ValueError(model2_file + ' does not exist. train.py did not do its job properly')

    # testing model1
    print('\npredicting localization')
    model1 = load_model(model1_file)
    print('model 1 loaded')
    IoU = predict_model1(
        folder_path_list=[actual_test_orig_folder_path, actual_test_loc_folder_path],
        color_mode_list=["rgb", "grayscale"],
        model=model1
    )
    print('IoU', IoU)
    print('localization prediction complete')

    print('\nclassification prediction')
    # model2 = load_model(model2_file)
    # print('model 2 loaded')

    masked_dir = os.path.join(data_folder_path, 'masked_data')
    # train.gen_masked_data(actual_test_orig_folder_path, actual_test_loc_folder_path, masked_dir)
    test_sample_num = train.get_train_sample_num(actual_test_orig_folder_path)
    print('\n#test images', test_sample_num)
    predict_inceptionv3(actual_test_orig_folder_path, model2_file, test_sample_num)

    if os.path.exists(masked_dir):
        rmtree(masked_dir)

    if os.path.exists(actual_test_orig_folder_path):
        rmtree(actual_test_orig_folder_path)

    if os.path.exists(actual_test_loc_folder_path):
        rmtree(actual_test_loc_folder_path)

    if os.path.exists(os.path.join(data_folder_path,test_suffix)):
        rmtree(os.path.join(data_folder_path,test_suffix))


if __name__ == "__main__":
    run()
