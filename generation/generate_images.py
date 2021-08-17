from torch.utils import data
import numpy as np
import  pytorch_fid_wrapper as pfw
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image
import math
import pickle

if __name__ == "__main__":
    device = 'cuda'
    # load data

    model_dir = './generators_weights/'
    img_dir = './generated_new_images/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    batch_size = 1
    dims = [64, 64]
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims)])
    print("Creating dataset object")
    dataset = datasets.ImageFolder(root='small-wikiart/', transform=transforms_compose)
    n_classes = len(dataset.classes)
    loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 120
    shared_dim = 128
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
    generator.load_state_dict(torch.load(model_dir+'gen.state_dict'))
    generator.eval()

    n_gen_images = 1000

    for y in range(n_classes):
        if not os.path.exists(img_dir+dataset.classes[y]+"/"):
            os.makedirs(img_dir+dataset.classes[y]+"/")
        print('class: {}'.format(dataset.classes[y]))
        y_emb = generator.shared_emb(torch.tensor([y]*batch_size, dtype=torch.long, device=device))
        for n_gen in range(n_gen_images):

            z = torch.clamp(torch.randn(batch_size, z_dim, device=device), min=-0.6, max=0.6)      # Generate random noise (z)
            fake = generator(z, y_emb)
            save_image(fake, img_dir+dataset.classes[y]+"/{}{}.png".format(dataset.classes[y],n_gen))