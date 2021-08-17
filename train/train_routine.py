from torch.utils import data
import numpy as np
import  pytorch_fid_wrapper as pfw
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image
import pickle

import sys
sys.path.append('./')
from models.ada_gan import Generator, Discriminator

class RandomApplyEach(nn.Module):
    def __init__(self, transforms, p):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        for t in self.transforms:
            if self.p > torch.rand(1, device='cuda'):
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ImageDatasetWrapper():
    def __init__(self, dataset):
        self.dataset=dataset

    def __getitem__(self, key):
        if isinstance(key, slice):
            range(*key.indices(len(self.dataset)))
            return torch.tensor([np.asarray(self.dataset[i][0]) for i in range(*key.indices(len(self.dataset)))])
        elif isinstance(key, int):
            return torch.tensor(self.dataset[key][0])

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    device = 'cuda'

    model_dir = './results/generators_weights/'
    img_dir = './results/generated_images/'
    var_dir = './results/saved_variables/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

    batch_size = 16
    dims = [64, 64]
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(dims)])
    print("Creating dataset object")
    # load data
    dataset = datasets.ImageFolder(root='./datasets/autsl/images/', transform=transforms_compose)
    n_classes = len(dataset.classes)
    loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)

    augmentation_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.5, 1.5)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ]
    p = torch.tensor(0.0, device=device)
    ada_target = 0.6
    update_iteration = 8
    adjustment_size = 500000 # number of images to reach p=1
    augmentation = RandomApplyEach(augmentation_transforms, p).to(device)
    ada_buf = torch.tensor([0.0, 0.0], device=device)

    # Initialize models
    print("Creating models")
    base_channels = 64
    z_dim = 120
    shared_dim = 128
    ortho_reg = False
    ortho_strength = 1e-4
    generator = Generator(base_channels=base_channels, bottom_width=8, z_dim=z_dim, shared_dim=shared_dim, n_classes=n_classes).to(device)
    discriminator = Discriminator(base_channels=base_channels, n_classes=n_classes).to(device)

    # Initialize weights orthogonally
    for module in generator.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            nn.init.orthogonal_(module.weight)
    for module in discriminator.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
            nn.init.orthogonal_(module.weight)

    # Initialize optimizers
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.999), eps=1e-6)

    # Setup FID
    print("Calculating FID parameters")
    pfw.set_config(batch_size=batch_size, device=device)
    if os.path.isfile(var_dir+'fid_stats.pkl'):
        with open(var_dir+'fid_stats.pkl', 'rb') as f:
            real_m, real_s = pickle.load(f)
    else:
        real_m, real_s = pfw.get_stats(ImageDatasetWrapper(dataset))
        with open(var_dir+'fid_stats.pkl', 'wb') as f:
            pickle.dump([real_m, real_s], f)
    print("FID parameters calculated!")

    ###########
    #  TRAIN  #
    ###########
    print("Training started")
    cur_step = 0
    min_fid = 2000
    e = 0
    D_steps = 2
    n_epochs = 100

    # Generate random noise (z)
    fixed_z = torch.randn(batch_size, z_dim, device=device)
    fixed_image, fixed_y = next(iter(loader))
    save_image(fixed_image, img_dir+"real.png")
    # Generate a batch of labels (y), one for each class
    fixed_y = fixed_y.to(device).long()
    # Retrieve class embeddings (y_emb) from generator
    fixed_y_emb = generator.shared_emb(fixed_y)

    image_number = 0

    fakes = torch.tensor([], device='cpu')
    fids = torch.tensor([], device='cpu')
    
    for epoch in range(n_epochs):
        print('##############################')
        print('#epoch: {}'.format(epoch))

        for batch_ndx, sample in enumerate(loader):
            real, labels = sample[0], sample[1]
            batch_size = len(real)
            real = real.to(device)
            real_augmented = augmentation(real)

            disc_opt.zero_grad()
            gen_opt.zero_grad()

            for i in range(D_steps):
                # Zero out the discriminator gradients
                disc_opt.zero_grad()
                ### Update discriminator ###
                # Get noise corresponding to the current batch_size 
                z = torch.randn(batch_size, z_dim, device=device)       # Generate random noise (z)
                y = labels.to(device).long()                            # Generate a batch of labels (y), one for each class
                y_emb = generator.shared_emb(y)                         # Retrieve class embeddings (y_emb) from generator
                fake = generator(z, y_emb)
                fake = augmentation(fake.detach())

                disc_fake_pred = discriminator(fake, y)
                disc_real_pred = discriminator(real_augmented, y)

                ada_buf += torch.tensor(
                    (torch.clamp(torch.sign(disc_real_pred), min=0, max=1).sum().item(), disc_real_pred.shape[0]),
                    device=device
                )

                #loss
                disc_loss = discriminator.loss(disc_fake_pred, disc_real_pred)
                # Update gradients
                disc_loss.backward()

                if ortho_reg:
                    ortho(discriminator, ortho_strength)

                # Update optimizer
                disc_opt.step()

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            fake = generator(z, y_emb)
            fake = augmentation(fake)
            disc_fake_pred = discriminator(fake, y)
            #loss
            gen_loss =  generator.loss(disc_fake_pred)
            # Update gradients
            gen_loss.backward()

            if ortho_reg:
                ortho(generator, ortho_strength, blacklist=[param for param in generator.shared_emb.parameters()])

            # Update optimizer
            gen_opt.step()

            fakes = torch.cat((fakes, fake.to('cpu')))

            if cur_step % update_iteration == 0:
                # Adaptive Data Augmentation
                pred_signs, n_pred = ada_buf
                r_t = pred_signs / n_pred

                sign = r_t - ada_target

                augmentation.p = torch.clamp(augmentation.p + (sign * n_pred / adjustment_size), min=0, max=1)

                ada_buf = ada_buf * 0

            cur_step +=1

            if cur_step % 1000 == 0:
                print('===========================================================================')
                val_fid = pfw.fid(fakes, real_m=real_m, real_s=real_s)
                fids = torch.cat((fids, torch.tensor([val_fid])))
                with open(var_dir+'fids.pkl', 'wb') as f:
                    pickle.dump(fids, f)
                with open(var_dir+'p.pkl', 'wb') as f:
                    pickle.dump(augmentation.p, f)
                fakes = torch.tensor([], device='cpu')
                print('FID: {}'.format(val_fid))
                print('augmentation p: {}'.format(augmentation.p))
                if (val_fid < min_fid):
                    min_fid = val_fid
                    save_image(fake, img_dir+"generated-with-better-FID{}.png".format(image_number))
                    torch.save(generator.state_dict(), (model_dir+'gen.state_dict'))
                    torch.save(discriminator.state_dict(), (model_dir+'disc.state_dict'))
                print('===========================================================================')

        print('saved images')
        fake = generator(fixed_z, fixed_y_emb)
        save_image(fake, img_dir+"generated{}.png".format(image_number))
        save_image(augmentation(fixed_image), img_dir+"augmented_real.png")
        image_number += 1