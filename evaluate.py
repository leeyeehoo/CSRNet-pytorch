import h5py
import scipy.io as sio
import PIL.Image as Image
import numpy as np
import os
import glob
import torchvision.transforms.functional as F
from image import *
from model import CSRNet
import torch
import time


from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


data_path='models/shanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/'
img_paths = glob.glob(os.path.join(data_path, '*.jpg'))


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


model = CSRNet(load_weights=True)

if use_cuda:
    model = model.cuda()
    checkpoint = torch.load('models/partBmodel_best.pth.tar')
else:
    checkpoint = torch.load('models/partBmodel_best.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])


mae = 0
for i in xrange(len(img_paths)):
    t1 = time.time()
    
    if use_cuda:
        img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    else:
        img = transform(Image.open(img_paths[i]).convert('RGB'))

    file_name = img_paths[i].replace('.jpg','.mat').replace('images','ground_truth')
    gt_file_name = os.path.join(os.path.dirname(file_name), 'GT_' + os.path.basename(file_name))
    
    gt_file = sio.loadmat(gt_file_name)
    #print(gt_file)
    #print(len(gt_file['image_info'][0][0][0][0][0]))
    #print(gt_file['image_info'][0][0][0][0][1][0][0])
    groundtruth = gt_file['image_info'][0][0][0][0][1][0][0]

    output = model(img.unsqueeze(0))
    t2 = time.time()
    print(t2-t1)
    print('{}  --  {} : {}'.format(img_paths[i], output.detach().cpu().sum().numpy(), groundtruth))
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    #print i,mae
print mae/len(img_paths)

