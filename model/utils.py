import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import glob
import re

import matplotlib.pyplot as plt


def read_split_data(faces_root: str, audio_root: str, val_rate: float = 0.2):
    random.seed(0) 

    assert os.path.exists(faces_root), "dataset root: {} does not exist.".format(faces_root)
    assert os.path.exists(audio_root), "dataset root: {} does not exist.".format(audio_root)


    face_class = [cla for cla in os.listdir(faces_root) if
                  os.path.isdir(os.path.join(faces_root, cla))]  # [fcmr0,fcrh0,fdac1,fdms0,fdrd1]
    audio_class = [cla for cla in os.listdir(audio_root) if
                   os.path.isdir(os.path.join(audio_root, cla))]  # [fcmr0,fcrh0,fdac1,fdms0,fdrd1]
 
    face_class.sort()
    audio_class.sort()
  
    class_indices = dict((k, v) for v, k in enumerate(face_class)) 
 
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = [] 
    train_images_label = [] 
    train_audio_path = []  
    train_audio_label = []  
    val_images_path = [] 
    val_images_label = []  
    val_audio_path = []  
    val_audio_label = []  

    supported = [".jpg", ".JPG", ".png", ".PNG"] 
    for cla in face_class:  
        faces_root2 = os.path.join(faces_root, cla)  # faces/fcmr0
        audio_root2 = os.path.join(audio_root, cla)  # audio/fcrm0
        faces_cla2_class = [cla2 for cla2 in os.listdir(faces_root2) if
                            os.path.isdir(os.path.join(faces_root2 + '/', cla2))]  # [sa1,sa2,...]
        for cla2 in faces_cla2_class:  
            faces_cla_path = os.path.join(faces_root2 + '/', cla2)  # faces/fcmr0/sa1
            faces_images = [os.path.join(faces_cla_path + '/', i) for i in os.listdir(faces_cla_path)
                            if os.path.splitext(i)[-1] in supported]
            audio_images = glob.glob('{}/{}**'.format(audio_root2, cla2))
            for i in audio_images:  
                if 'chunk' not in i:
                    audio_images.remove(i)

            audio_images = sorted(audio_images, key=lambda i: int(re.findall(r'\d+', i)[-1]))

            fl = len(faces_images)
            al = len(audio_images)
            if fl > al:
                faces_images = faces_images[:al]
            elif fl < al:
                audio_images = audio_images[:fl]

            image_class = class_indices[cla]

            val_f_path = random.sample(faces_images, k=int(len(faces_images) * val_rate))
            # val_a_path = random.sample(audio_images, k=int(len(audio_images) * val_rate))
            val_a_path = []
            for i in val_f_path:
                index = faces_images.index(i)
                val_a_path.append(audio_images[index])

            for img_path in faces_images:
                if img_path in val_f_path: 
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:  
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)

            for img_path in audio_images:
                if img_path in val_a_path:  
                    val_audio_path.append(img_path)
                    val_audio_label.append(image_class)
                else:
                    train_audio_path.append(img_path)
                    train_audio_label.append(image_class)

    print("{} images were found in the dataset.".format(
        len(train_images_path) + len(val_images_path) + len(train_audio_path) + len(val_audio_path)))
    print("{} images for training.".format(len(train_images_path) + len(train_audio_path)))
    print("{} images for validation.".format(len(val_images_path) + len(val_audio_path)))
    return train_images_path, train_images_label, train_audio_path, train_audio_label, val_images_path, val_images_label, val_audio_path, val_audio_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = ''
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([]) 
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(castModel, model, optimizer1, optimizer2, data_loader, device, epoch):
    model.train()
    castModel.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, images_data in enumerate(data_loader):
        images, images_labels, audio, audio_labels = images_data 

        sample_num = sample_num + images.shape[0]

        for i in range(2):
            optimizer1.zero_grad()
            castLoss, pred = castModel(images.to(device), audio.to(device))
            castLoss.backward(retain_graph=True)
            optimizer1.step()


        pred1 = model(pred.detach())
        pred_classes = torch.max(pred1, dim=1)[1]
        accu_num = accu_num + torch.eq(pred_classes, images_labels.to(device)).sum()  # ??????

        optimizer2.zero_grad()
        loss = loss_function(pred1, images_labels.to(device))
        loss.backward()
        accu_loss = accu_loss + loss.detach()
        optimizer2.step()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)



    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(castModel, model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    castModel.eval()

    accu_num = torch.zeros(1).to(device) 
    accu_loss = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, images_data in enumerate(data_loader):
        images, images_labels, audio, audio_labels = images_data
        sample_num = sample_num + images.shape[0]

        _, pred = castModel(images.to(device), audio.to(device))
        pred1 = model(pred)
        pred_classes = torch.max(pred1, dim=1)[1]
        accu_num = accu_num + torch.eq(pred_classes, images_labels.to(device)).sum()

        loss = loss_function(pred1, images_labels.to(device))
        accu_loss = accu_loss + loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
