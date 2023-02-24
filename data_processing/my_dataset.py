from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, audio_path: list, audio_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.audio_path = audio_path
        self.audio_class = audio_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        audio = Image.open(self.audio_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if audio.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.audio_path[item]))

        label_images = self.images_class[item]
        label_audio = self.audio_class[item]

        if self.transform is not None:
            img = self.transform(img)
            audio = self.transform(audio)

        return img, label_images, audio, label_audio

    @staticmethod
    def collate_fn(batch):
        images, labels_images, audio, labels_audio = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        audio = torch.stack(audio, dim=0)
        labels_images = torch.as_tensor(labels_images)
        labels_audio = torch.as_tensor(labels_audio)
        return images, labels_images, audio, labels_audio