import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
                    'progan_with_post-processing': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/proganwithnoise',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },
                # 'seeingdark': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/seeingdark',
                #                  'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                #                  'no_crop'    : True,
                #                },
                
        #         'ForenSynths': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/ForenSynths/',
        #                          'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        #                          'no_crop'    : True,
        #                        },
        #    'GANGen-Detection': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/GANGen-Detection/',
        #                          'no_resize'  : True,
        #                          'no_crop'    : True,
        #                        },

        #  'DiffusionForensics': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/DiffusionForensics/',
        #                          'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        #                          'no_crop'    : True,
        #                        },

        # 'UniversalFakeDetect': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/UniversalFakeDetect/',
        #                          'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        #                          'no_crop'    : True,
        #                        },
        #         'Diffusion1kStep': { 'dataroot'   : '/home/fanzheming/zm/NPR-DeepfakeDetection/dataset/Diffusion1kStep/',
        #                          'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        #                          'no_crop'    : True,
        #                        },

                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
model = resnet50(num_classes=1)
# model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)

try:
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
except:
    from collections import OrderedDict
    from copy import deepcopy
    state_dict = torch.load(opt.model_path, map_location='cpu')['model']
    pretrained_dict = OrderedDict()
    for ki in state_dict.keys():
        pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
    model.load_state_dict(pretrained_dict, strict=True)

model.cuda()
model.eval()

# for testSet in DetectionTests.keys():
#     dataroot = DetectionTests[testSet]['dataroot']
#     printSet(testSet)

#     accs = [];aps = []
#     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
#     for v_id, val in enumerate(os.listdir(dataroot)):
#         opt.dataroot = '{}/{}'.format(dataroot, val)
#         opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
#         opt.no_resize = DetectionTests[testSet]['no_resize']
#         opt.no_crop   = DetectionTests[testSet]['no_crop']
#         acc, ap, _, _, _, _ = validate(model, opt)
#         accs.append(acc);aps.append(ap)
#         print("({} {:12}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, acc*100, ap*100))
#     print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = []
    aps = []
    fnrs = []
    fprs = []
    raccs = []  # 存储 r_acc 的列表
    faccs = []  # 存储 f_acc 的列表
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    
    for v_id, val in enumerate(sorted(os.listdir(dataroot))):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes = ''  # os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop = DetectionTests[testSet]['no_crop']
        
        acc, ap, r_acc, f_acc, fnr, fpr, _, _ = validate(model, opt)

        accs.append(acc)
        aps.append(ap)
        fnrs.append(fnr)
        fprs.append(fpr)
        raccs.append(r_acc)  # 添加 r_acc
        faccs.append(f_acc)  # 添加 f_acc

        # 输出样本名称和 v_id
        print(f"({v_id} {val})")
        
        # 输出其他指标，设置固定宽度
        print("  acc: {0:6.2f};  ap: {1:6.2f};  fnr: {2:6.2f};  fpr: {3:6.2f};  r_acc: {4:6.2f};  f_acc: {5:6.2f}".format(
            acc * 100, ap * 100, fnr * 100, fpr * 100, r_acc * 100, f_acc * 100))

    # 输出均值
    print("Mean:")  # 添加换行
    print("  acc: {0:6.2f};  ap: {1:6.2f};  fnr: {2:6.2f};  fpr: {3:6.2f};  r_acc: {4:6.2f};  f_acc: {5:6.2f}".format(
        np.array(accs).mean() * 100, np.array(aps).mean() * 100,
        np.array(fnrs).mean() * 100, np.array(fprs).mean() * 100,
        np.array(raccs).mean() * 100, np.array(faccs).mean() * 100))
    
    print('*' * 25)




