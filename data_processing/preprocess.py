import os
import json
import random
import glob


def read_split_data(video_root: str, audio_root: str, val_rate: float = 0.2):
    random.seed(0)

    assert os.path.exists(video_root), "dataset root: {} does not exist.".format(video_root)
    assert os.path.exists(audio_root), "dataset root: {} does not exist.".format(audio_root)
    # Get audio and video paths
    video_class = [cla for cla in os.listdir(video_root) if
                   os.path.isdir(os.path.join(video_root, cla))]
    audio_class = [cla for cla in os.listdir(audio_root) if
                   os.path.isdir(os.path.join(audio_root, cla))]
    video_class.sort()
    audio_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(video_class))
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
    for cla in video_class:
        video_root2 = os.path.join(video_root, cla)
        audio_root2 = os.path.join(audio_root, cla)
        video_cla2_class = [cla2 for cla2 in os.listdir(video_root2) if
                            os.path.isdir(os.path.join(video_root2 + '/', cla2))]
        for cla2 in video_cla2_class:
            video_cla_path = os.path.join(video_root2 + '/', cla2)
            video_images = [os.path.join(video_cla_path + '/', i) for i in os.listdir(video_cla_path)
                            if os.path.splitext(i)[-1] in supported]
            audio_images = glob.glob('{}/{}**'.format(audio_root2, cla2))
            for i in audio_images:
                if 'chunk' not in i:
                    audio_images.remove(i)
            audio_images = sorted(audio_images, key=lambda i: int(re.findall(r'\d+', i)[-1]))

            fl = len(video_images)
            al = len(audio_images)
            if fl > al:
                video_images = video_images[:al]
            elif fl < al:
                audio_images = audio_images[:fl]

            image_class = class_indices[cla]

            val_f_path = random.sample(video_images, k=int(len(video_images) * val_rate))
            val_a_path = []
            for i in val_f_path:
                index = video_images.index(i)
                val_a_path.append(audio_images[index])

            for img_path in video_images:
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
