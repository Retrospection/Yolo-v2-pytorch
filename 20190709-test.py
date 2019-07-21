# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from inference import FeatureExtractor

WEBFACE_ROOT = 'D:\\dev\\dataset\\CASIA-WebFace'

raw_register_lists = [line.strip() for line in open('data\\webface\\register.txt')]
raw_test_lists = [line.strip() for line in open('data\\webface\\register.txt')]
raw_register_label = [os.path.split(p)[0] for p in raw_register_lists]
raw_test_label = [os.path.split(p)[0] for p in raw_test_lists]

fe = FeatureExtractor()

register_features = [fe.get_feature(os.path.join(WEBFACE_ROOT, p)) for p in raw_register_lists]
register_features = np.vstack(register_features)
test_features = [fe.get_feature(os.path.join(WEBFACE_ROOT, p)) for p in raw_register_lists]


right = 0
error = 0
for test_feature, label in zip(test_features, raw_test_label):
    index = np.argmin(np.linalg.norm(test_feature - register_features))
    if raw_register_label[index] == label:
        right += 1
    else:
        error += 1

open('result.txt', 'w').write("""
right: {}
error: {}
rate: {}
""".format(right,error, right / (right + error)))



######################################################################
# 以下构建测试数据集
######################################################################


# import pathlib
# import random
#
# images = []
#
# register_annotation = open('data\\webface\\register.txt', 'w')
# test_annotation = open('data\\webface\\test.txt', 'w')
#
# for root, person_ids, _ in os.walk(WEBFACE_ROOT):
#
#     chosen_persons = random.sample(person_ids, 100)
#     for chosen_person in chosen_persons:
#         for person_root, _, filenames in os.walk(os.path.join(root, chosen_person)):
#             num_of_files = len(filenames)
#             num_of_register_files = (num_of_files // 3) * 2
#             register_files = random.sample(filenames, num_of_register_files)
#             test_files = list(set(filenames) - set(register_files))
#             serialized_register_filenames = [os.path.join(chosen_person, filename) + '\n' for filename in register_files]
#             serialized_test_filenames = [os.path.join(chosen_person, filename) + '\n' for filename in test_files]
#             register_annotation.writelines(serialized_register_filenames)
#             test_annotation.writelines(serialized_test_filenames)
#     break






