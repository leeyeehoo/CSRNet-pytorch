import os
import glob
import h5py
import scipy.io as io
import cv2
import matplotlib.pyplot as plt
from image import *
from scipy.ndimage.filters import gaussian_filter


# Set the root to the Shanghai dataset you download
ROOT = '/Users/kshitiz/Documents/GitHub/CSRNet-pytorch/ShanghaiTech'

def get_img_paths(root, train_path, test_path):
    """
    return the image paths
    """
    part_a_train = os.path.join(root, train_path, 'images')
    part_a_test = os.path.join(root, test_path, 'images')
    path_sets = [part_a_train, part_a_test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    return img_paths


def process_images(img_paths):
    """
    Resize and generate the ground truth
    """
    for img_path in img_paths:
        print(img_path)
        image = cv2.imread(img_path)

        # Resize image dimensions
        d_width, d_height = 640, 360 # setup the image dimension 
        re_image = cv2.resize(image, (d_width, d_height), cv2.INTER_AREA)
        cv2.imwrite(img_path, re_image)

        # Load corresponding mat file
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('DSC_', 'GT_DSC_').replace('20221212_', 'GT_20221212_'))
        img = plt.imread(img_path)

        # Prepare empty density map
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]

        # calculate the scaling factor for x and y dim
        scale_x = d_width / image.shape[1]
        scale_y = d_height / image.shape[0]
        for i in range(len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                # scaling to new image dimension
                gt[i][1] = scale_x * gt[i][1]
                gt[i][0] = scale_y * gt[i][0]
                k[int(gt[i][1]), int(gt[i][0])] = 1 # mark with 1 to indicate presence at the location

        k = gaussian_filter(k, 15)
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf: # Save as .h5
            hf['density'] = k

def train_test_path(root_path, train_data_folder, test_data_folder):
    train_images = [os.path.join(root_path, train_data_folder, img) for img in os.listdir(os.path.join(root_path, train_data_folder)) if img.endswith('.jpg')]
    test_images = [os.path.join(root_path, test_data_folder, img) for img in os.listdir(os.path.join(root_path, test_data_folder)) if img.endswith('.jpg')]

    return train_images + test_images
def obtain_images(root, train_data_folder, test_data_folder):
    image_paths = train_test_path(root, train_data_folder, test_data_folder)
    process_images(image_paths)

if __name__ == '__main__':
    ROOT = '/Users/kshitiz/Documents/GitHub/CSRNet-pytorch/ShanghaiTech'
    TRAIN_DATA_FOLDER, TEST_DATA_FOLDER = 'train_data', 'test_data'
    obtain_images(ROOT, TRAIN_DATA_FOLDER, TEST_DATA_FOLDER)
    