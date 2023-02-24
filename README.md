# AVoiD-DF
## This repository contains code and models for our paper
> AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake
> > In this paper, we propose an Audio-Visual Joint Learning for Detecting Deepfake (AVoiD-DF), which exploits audio-visual inconsistency for multi-modal forgery detection.

Clone the repo:
'''
git clone https://github.com/SYSU-DISG/AVoiD-DF.git
'''

Download the dataset and weights from the following link:
'''
https://pan.baidu.com/s/1MckHs-H57jTma5v0o6XYMA
'''

We use Python 3.6. Install requirements by running:
'''
pip install -r requirements.txt
'''

We create our own dataloader found in 'data_processing/my_dataset', you can modify it to train with your own data.

Training
You can do training on your own data by a simple command:
'''
python train.py
'''

Acknowledgements
Our work is based on the official version of AVoiD-DF, and some of our codes refer to VIT (Vision Transformer). Thanks for sharing!

Citation
If you find our repo helpful to your research, please cite our paper, thanks.
