# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torchvision.transforms as T
from PIL import Image

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('logs/tmp/models/best_model.pth')
state_dict = checkpoint['state_dict']
label2name = checkpoint['label2name']
network = get_resnet50(len(label2name), pretrain=False)
network.load_state_dict(state_dict)
network = network.to(device)
network.eval()

resize = T.Resize((224, 224))
totensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


def classify_plant(image):
    image = resize(image)
    image = totensor(image)
    image = normalize(image)
    image = image.unsqueeze(0)  # add an axial to [1, 3, 224, 224]
    image = image.to(device)
    with torch.no_grad():
        scores = network(image)
    pred_label = label2name[scores.max(1)[1].item()]
    scores = scores.cpu().numpy()
    return pred_label, scores


if __name__ == '__main__':
    img_path = ''
    origin_image = Image.open(img_path).convert("RGB")

    pred_label, scores = classify_plant(origin_image)
