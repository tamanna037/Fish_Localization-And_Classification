import os
import glob
import sys

import numpy as np
from scipy.misc import imread, imsave, imresize
import json
import shutil
import cv2
import zipfile
from shutil import make_archive, copy2, rmtree

# Image processing
IMAGE_ROW_SIZE = 145
IMAGE_COLUMN_SIZE = 145

# Training and Testing procedure
PATIENCE = 4
BATCH_SIZE = 32
INSPECT_SIZE = 4

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def load_annotation(ANNOTATION_FOLDER_PATH):
    annotation_dict = {}
    # The glob module finds all the pathnames matching a specified pattern
    annotation_file_path_list = glob.glob(os.path.join(ANNOTATION_FOLDER_PATH, "*.json"))
    for annotation_file_path in annotation_file_path_list:
        print(annotation_file_path)
        with open(annotation_file_path) as annotation_file:
            annotation_file_content = json.load(annotation_file)
            for item in annotation_file_content:
                key = os.path.basename(item["filename"])
                if key in annotation_dict:
                    assert False, "Found existing key {}!!!".format(key)
                value = []
                for annotation in item["annotations"]:
                    value.append(
                        np.clip((annotation["x"], annotation["width"], annotation["y"], annotation["height"]), 0,
                                np.inf).astype(np.int))
                annotation_dict[key] = value
    return annotation_dict


def reformat_localization(LOCALIZATION_FOLDER_PATH, TRAIN_FOLDER_PATH, annotations_folder_path):
    print("Creating the localization folder ...")
    os.makedirs(LOCALIZATION_FOLDER_PATH, exist_ok=True)

    print("Loading annotation ...")
    annotation_dict = load_annotation(annotations_folder_path)

    original_image_path_list = glob.glob(os.path.join(TRAIN_FOLDER_PATH, "*/*"))
    for fish in FishNames:
        os.makedirs(LOCALIZATION_FOLDER_PATH + '/' + fish, exist_ok=True)
    for original_image_path in original_image_path_list:
        localization_image_path = LOCALIZATION_FOLDER_PATH + original_image_path[len(TRAIN_FOLDER_PATH):]
        if os.path.isfile(localization_image_path):
            continue

        localization_image_content = np.zeros(imread(original_image_path).shape[:2], dtype=np.uint8)
        for annotation_x, annotation_width, annotation_y, annotation_height in annotation_dict.get(
                os.path.basename(original_image_path), []):
            localization_image_content[annotation_y:annotation_y + annotation_height,
            annotation_x:annotation_x + annotation_width] = 255

        # os.makedirs(os.path.abspath(os.path.join(localization_image_path, os.pardir)), exist_ok=True)
        imsave(localization_image_path, localization_image_content)


def create_80_10_10_split(data_folder_path):
    MAX_IMAGES_PER_SAMPLE = 200
    root_train_path = os.path.join(data_folder_path, 'train_data')
    root_loc_path = os.path.join(data_folder_path, 'localization')

    train_folder_path_1 = os.path.join(data_folder_path, 'Train', 'Best_hyperparameter_80_percent')
    val_folder_path_1 = os.path.join(data_folder_path, 'Validation', 'Validation_10_percent')
    test_folder_path_1 = os.path.join(data_folder_path, 'Test', 'Test_10_percent')
    os.makedirs(train_folder_path_1, exist_ok=True)
    os.makedirs(val_folder_path_1, exist_ok=True)
    os.makedirs(test_folder_path_1, exist_ok=True)

    train_loc_folder_path = os.path.join(train_folder_path_1, 'train_localization')
    train_folder_path = os.path.join(train_folder_path_1, 'train_original')
    val_loc_folder_path = os.path.join(val_folder_path_1, 'valid_localization')
    val_folder_path = os.path.join(val_folder_path_1, 'valid_original')
    test_loc_folder_path = os.path.join(test_folder_path_1, 'test_localization')
    test_folder_path = os.path.join(test_folder_path_1, 'test_original')

    for fish in FishNames:
        print('class ', fish, 'processing')
        if fish not in os.listdir(root_train_path):
            os.mkdir(os.path.join(root_train_path, fish))
        if fish not in os.listdir(root_loc_path):
            os.mkdir(os.path.join(root_loc_path, fish))

        total_images = os.listdir(os.path.join(root_train_path, fish))
        total = min(len(total_images), MAX_IMAGES_PER_SAMPLE)
        total_images = total_images[:total]
        # np.random.shuffle(total_images)

        nbr_train = int(len(total_images) * 0.8)
        train_images = total_images[:nbr_train]
        print('train length', len(train_images))
        os.makedirs(os.path.join(train_folder_path, fish), exist_ok=True)
        os.makedirs(os.path.join(train_loc_folder_path, fish), exist_ok=True)
        for img in train_images:
            source = os.path.join(root_train_path, fish, img)
            target = os.path.join(train_folder_path, fish, img)
            shutil.copy(source, target)
            source = os.path.join(root_loc_path, fish, img)
            target = os.path.join(train_loc_folder_path, fish, img)
            shutil.copy(source, target)

        nbr_val = int(len(total_images) * 0.1)
        val_images = total_images[nbr_train:nbr_train + nbr_val]
        print('val length', len(val_images))
        os.makedirs(os.path.join(val_folder_path, fish), exist_ok=True)
        os.makedirs(os.path.join(val_loc_folder_path, fish), exist_ok=True)
        for img in val_images:
            source = os.path.join(root_train_path, fish, img)
            target = os.path.join(val_folder_path, fish, img)
            shutil.copy(source, target)
            source = os.path.join(root_loc_path, fish, img)
            target = os.path.join(val_loc_folder_path, fish, img)
            shutil.copy(source, target)

        nbr_test = int(len(total_images) * 0.1)
        test_images = total_images[nbr_train + nbr_val:nbr_train + nbr_val + nbr_test]
        print('test length', len(test_images))
        os.makedirs(os.path.join(test_folder_path, fish), exist_ok=True)
        os.makedirs(os.path.join(test_loc_folder_path, fish), exist_ok=True)
        for img in test_images:
            source = os.path.join(root_train_path, fish, img)
            target = os.path.join(test_folder_path, fish, img)
            shutil.copy(source, target)
            source = os.path.join(root_loc_path, fish, img)
            target = os.path.join(test_loc_folder_path, fish, img)
            shutil.copy(source, target)
    print('created 80-10-10 split\n')
    zipit([train_folder_path, train_loc_folder_path], os.path.join(train_folder_path_1, 'data.zip'))
    zipit([val_folder_path, val_loc_folder_path], os.path.join(val_folder_path_1, 'data.zip'))
    zipit([test_folder_path, test_loc_folder_path], os.path.join(test_folder_path_1, 'data.zip'))
    rmtree(train_folder_path)
    rmtree(train_loc_folder_path)
    rmtree(val_folder_path)
    rmtree(val_loc_folder_path)
    rmtree(test_folder_path)
    rmtree(test_loc_folder_path)


def create_10_or_90_min_split(data_folder_path, MAX_IMAGES_PER_SAMPLE, path):
    root_train_path = os.path.join(data_folder_path, 'train_data')
    root_loc_path = os.path.join(data_folder_path, 'localization')

    train_folder_path_1 = os.path.join(data_folder_path, 'Train', path)
    os.makedirs(train_folder_path_1, exist_ok=True)

    train_loc_folder_path = os.path.join(train_folder_path_1, 'train_original')
    train_folder_path = os.path.join(train_folder_path_1, 'train_localization')

    for fish in FishNames:
        print('class ', fish, 'processing')
        if fish not in os.listdir(root_train_path):
            os.mkdir(os.path.join(root_train_path, fish))
        if fish not in os.listdir(root_loc_path):
            os.mkdir(os.path.join(root_loc_path, fish))

        total_images = os.listdir(os.path.join(root_train_path, fish))
        total = min(len(total_images), MAX_IMAGES_PER_SAMPLE)
        train_images = total_images[:total]
        # np.random.shuffle(total_images)

        os.makedirs(os.path.join(train_folder_path, fish), exist_ok=True)
        os.makedirs(os.path.join(train_loc_folder_path, fish), exist_ok=True)
        for img in train_images:
            source = os.path.join(root_train_path, fish, img)
            target = os.path.join(train_folder_path, fish, img)
            shutil.copy(source, target)
            source = os.path.join(root_loc_path, fish, img)
            target = os.path.join(train_loc_folder_path, fish, img)
            shutil.copy(source, target)

    print('created ' + path + '\n')
    zipit([train_folder_path, train_loc_folder_path], os.path.join(train_folder_path_1, 'data.zip'))
    rmtree(train_folder_path)
    rmtree(train_loc_folder_path)


def create_3_sample_set(data_folder_path):
    MAX_IMAGES_PER_SAMPLE = 3
    root_train_path = os.path.join(data_folder_path, 'train_data')
    root_loc_path = os.path.join(data_folder_path, 'localization')

    val_folder_path_1 = os.path.join(data_folder_path, 'Validation', '3_samples')
    os.makedirs(val_folder_path_1, exist_ok=True)

    val_loc_folder_path = os.path.join(val_folder_path_1, 'train_original')
    val_folder_path = os.path.join(val_folder_path_1, 'train_localization')

    for fish in FishNames:
        print('class ', fish, 'processing')
        if fish not in os.listdir(root_train_path):
            os.mkdir(os.path.join(root_train_path, fish))
        if fish not in os.listdir(root_loc_path):
            os.mkdir(os.path.join(root_loc_path, fish))

        total_images = os.listdir(os.path.join(root_train_path, fish))
        total = min(len(total_images), MAX_IMAGES_PER_SAMPLE)
        train_images = total_images[:total]
        # np.random.shuffle(total_images)

        os.makedirs(os.path.join(val_folder_path, fish), exist_ok=True)
        os.makedirs(os.path.join(val_loc_folder_path, fish), exist_ok=True)
        for img in train_images:
            source = os.path.join(root_train_path, fish, img)
            target = os.path.join(val_folder_path, fish, img)
            shutil.copy(source, target)
            source = os.path.join(root_loc_path, fish, img)
            target = os.path.join(val_loc_folder_path, fish, img)
            shutil.copy(source, target)

    print('created 3-sample-set\n')
    zipit([val_folder_path, val_loc_folder_path], os.path.join(val_folder_path_1, 'data.zip'))
    rmtree(val_folder_path)
    rmtree(val_loc_folder_path)


def convert_annotation_to_localization(annotation_array, row_size=IMAGE_ROW_SIZE, column_size=IMAGE_COLUMN_SIZE):
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
                cv2.resize(masked_image, (IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE))
                cv2.imwrite(masked_path, masked_image)
    print('masked data generated')


def zipit(folders, zip_filename):
    print('zipfilename', zip_filename)
    zip_file = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            # print(dirpath, dirnames, filenames)
            for filename in filenames:
                # relpath = os.path.relpath(os.path.join(dirpath, filename), os.path.join(folders[0], '../..')).split(sep='\\')
                # relpath = os.path.join(relpath[1],relpath[2])
                # print(relpath)
                zip_file.write(
                    os.path.join(dirpath, filename),
                    os.path.relpath(os.path.join(dirpath, filename), os.path.join(folders[0], '../..')))

    zip_file.close()


def unzip_files(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()


def run():
    if len(sys.argv) != 2:
        raise ValueError(
            'expected 2 arguments.'
            + "\neg. python.exe data.py .\Data\data.zip")
    data_file_path = sys.argv[1]
    data_folder_path = os.path.dirname(data_file_path)
    unzip_files(data_file_path, data_folder_path)

    traindata_folder_path = os.path.join(data_folder_path, 'train_data')
    annotations_folder_path = os.path.join(data_folder_path, 'annotations')
	
	## nicher duita ken exist korbe?
	
    if not os.path.exists(traindata_folder_path):
        raise ValueError(traindata_folder_path + ' does not exist. data.py did not do its job properly')
    if not os.path.exists(annotations_folder_path):
        raise ValueError(annotations_folder_path + ' does not exist. data.py did not do its job properly')

    localization_folder_path = os.path.join(data_folder_path, 'localization')
    print("Reformatting localization ...")
    reformat_localization(localization_folder_path, traindata_folder_path, annotations_folder_path)

    os.makedirs(os.path.join(data_folder_path, 'Train'), exist_ok=True)
    os.makedirs(os.path.join(data_folder_path, 'Validation'), exist_ok=True)
    os.makedirs(os.path.join(data_folder_path, 'Test'), exist_ok=True)
    create_80_10_10_split(data_folder_path)
    create_10_or_90_min_split(data_folder_path, 2, 'Under_10_min_training')
    create_10_or_90_min_split(data_folder_path, 8, 'Under_90_min_tuning')
    create_3_sample_set(data_folder_path)

    if os.path.exists(localization_folder_path):
        rmtree(localization_folder_path)
    if os.path.exists(traindata_folder_path):
        rmtree(traindata_folder_path)
    if os.path.exists(annotations_folder_path):
        rmtree(annotations_folder_path)


if __name__ == "__main__":
    run()
