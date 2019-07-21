# coding: utf-8

import re
import pickle

identityFilePath = r'D:\Program\essay\code\Yolo-v2-pytorch\data\celeba\annotations\identity_CelebA.txt'
bboxFilePath = r'D:\Program\essay\code\Yolo-v2-pytorch\data\celeba\annotations\list_bbox_celeba.txt'

identities = {}
bboxes = {}

with open(identityFilePath) as f:
    for line in f:
        filename, label = line.strip().split(' ')
        identities[filename.split('.')[0]] = label

with open(bboxFilePath) as f:
    for line in f:
        filename, x, y, w, h = re.split(' +', line.strip())
        bboxes[filename.split('.')[0]] = [[int(x), int(y), int(w), int(h), int(identities[filename.split('.')[0]])]]


annotation = {}

for filename, label in identities.items():
    annotation[filename] = {
        'file_name': str(filename) + '.jpg',
        'objects': bboxes[filename]
    }

pickle.dump(annotation, open(r'D:\Program\essay\code\Yolo-v2-pytorch\data\celeba\anno_pickle\celeba_total.pkl', 'wb'))
