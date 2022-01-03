import os
import cv2
import numpy as np

def get_image_list(path):
    image_list = []
    for (root, dirs, files) in os.walk(path):
        for fname in files:
            if os.path.splitext(os.path.basename(fname))[-1] in ['.jpeg', '.jpg', '.bmp', '.png', '.tif']:
                image_list.append(os.path.join(root, fname))
    return image_list

def make_test_label(test_imgdir, test_labdir, test_imgpath, img_size):
    # test_imgdir : bottle/test
    # test_labdir : bottle/ground_truth
    # test_imgpath : bottle/test/broken_large/000.png ||| bottle/test/good/000.png
    # test_labpath : bottle/ground_truth/broken_large/000_mask.png
    test_labpath_base = test_imgpath.replace(test_imgdir, test_labdir)
    test_labpath = os.path.join(test_labpath_base.split('.png')[0] + '_mask.png')
    if os.path.exists(test_labpath):
        label = cv2.resize(cv2.imread(test_labpath, cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_NEAREST)
        label[label>=1] = 1
        return (label, 1)
    else:
        print('label not exists: ', test_imgpath)
        return (np.zeros(shape=img_size) , 0)

def get_basename(path):
    return os.path.basename(os.path.dirname(path))+'_'+os.path.splitext(os.path.basename(path))[0]