import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import cv2
import torch


def to_rgb(known_channel, unknown_channels, color_space):
    """
    Takes a batch of images
    """
    if color_space == 'Lab':
        L = (known_channel + 1.) * 50.
        ab = unknown_channels * 110.
        Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []
        for img in Lab:
            img_rgb = lab2rgb(img)
            rgb_imgs.append(img_rgb)
        return np.stack(rgb_imgs, axis=0)
    elif color_space == 'HSL':
        L = ((known_channel + 1.) / 2.) * 255.
        H = (unknown_channels[:, 0] + 1.) * 90.
        S = ((unknown_channels[:, 1] + 1.) / 2.) * 255.
        dim = H.size()
        H = torch.reshape(H, (dim[0], 1, dim[1], dim[2]))
        S = torch.reshape(S, (dim[0], 1, dim[1], dim[2]))

        HLS = torch.cat([H, L, S], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []
        for img in HLS:
            # print('hls')
            # print(img)
            img_rgb = cv2.cvtColor(np.uint8(img), cv2.COLOR_HLS2RGB)
            # print('rgb')
            # print(img_rgb)
            rgb_imgs.append(img_rgb)
        return np.stack(rgb_imgs, axis=0)
    
def visualize(model, data, color_space, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.unknown_channels
    known_channel = model.known_channel
    fake_imgs = to_rgb(known_channel, fake_color, color_space)
    real_imgs = to_rgb(known_channel, real_color, color_space)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(known_channel[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")