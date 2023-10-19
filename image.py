import random
import h5py
import numpy as np
import cv2
from PIL import Image

def load_data(img_path, train=True):
    """
    loads the image and corresponding density map and returns the tuple (image & density map)
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')

    img = Image.open(img_path).convert('RGB') # load image and ground truth
    with h5py.File(gt_path, 'r') as gt_file:
        target = np.asarray(gt_file['density'])
    # allows for image augmentation
    if False:
        crop_size = (img.size[0] // 2, img.size[1] // 2)

        if random.randint(0, 9) <= -1:
            dx = random.randint(0, 1) * img.size[0] // 2
            dy = random.randint(0, 1) * img.size[1] // 2
        else:
            dx = int(random.random() * img.size[0] * 0.5)
            dy = int(random.random() * img.size[1] * 0.5)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1]+dy, dx:crop_size[0]+dx]

        # horizontal flip
        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Resizing the density map
    target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return img, target
