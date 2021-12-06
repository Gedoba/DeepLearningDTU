import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, ycbcr2rgb
import cv2
import torch
from copy import deepcopy
from torchvision import transforms
from PIL import Image
from data_generator import make_dataloaders

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
    elif color_space == 'YCbCr':
        Y = ((known_channel + 1.) * 219.) /2. + 16.
        CbCr = ((unknown_channels + 1.) * 112.) + 16.
        YCbCr = torch.cat([Y, CbCr], dim=1).permute(0, 2, 3, 1).cpu().numpy()
        rgb_imgs = []
        for img in YCbCr:
            img_rgb = ycbcr2rgb(img)
            rgb_imgs.append(img_rgb)
        return np.stack(rgb_imgs, axis=0)


def resize_np_array_image(img, height, width, grey_scale = False):
    if height is not None and width is not None:
        if grey_scale: # then img is Tensor not ndarray
            img = img.numpy()
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img

    
def visualize(model, data, color_space, save=True, width = None, height = None, model_name = None, set_num = 0):
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
    for i in range(known_channel.size(0)):
        ax = plt.subplot(3, 5, i + 1)
        gray_img = known_channel[i][0].cpu()
        gray_img = resize_np_array_image(gray_img, height, width, True)
        ax.imshow(gray_img, cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        fake_img = fake_imgs[i]
        fake_img = resize_np_array_image(fake_img, height, width)
        ax.imshow(fake_img)
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        real_img = real_imgs[i]
        real_img = resize_np_array_image(real_img, height, width)
        ax.imshow(real_img)
        ax.axis("off")
    plt.show()
    if save:
        file_name = "colorization_vis_"
        if model_name is not None:
            file_name += model_name
        else:
            file_name += color_space
        file_name += "_" + str(set_num) + ".png"
        fig.savefig(f"./visualizations/{file_name}")


def plot_metrics(iterations, loss_meter_dict, title = "Training Loss", save_path = None):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.plot(iterations, loss_meter_dict["loss_D_fake"].values, label='loss_D_fake')
    plt.plot(iterations, loss_meter_dict["loss_D_real"].values, label='loss_D_real')
    plt.plot(iterations, loss_meter_dict["loss_D"].values, label='loss_D')
    plt.plot(iterations, loss_meter_dict["loss_G_GAN"].values, label='loss_G_GAN')
    plt.plot(iterations, loss_meter_dict["loss_G_L1"].values, label='loss_G_L1')
    plt.plot(iterations, loss_meter_dict["loss_G"].values, label='loss_G')
    plt.legend()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path)


def get_plot_save_path(id, path):
    return None if path is None else f"{path}_{id}.png"


def plot_all_metrics(loss_meter_dict, train_loss_meter_dict, path = None):
    epochs_count = len(loss_meter_dict['loss_G_GAN'].values)
    iters_count = len(train_loss_meter_dict['loss_G_GAN'].values)
    plot_metrics(range(1, epochs_count + 1), loss_meter_dict, f"Validation loss (1-{epochs_count} epochs)", get_plot_save_path("val_loss", path))
    plot_metrics(range(1, iters_count + 1), train_loss_meter_dict, f"Training loss (1-{epochs_count} epochs)", get_plot_save_path("train_loss", path))

    iters_per_epoch = int(iters_count / epochs_count)

    copied_train_loss_meter_dict = deepcopy(train_loss_meter_dict)
    for loss_name, loss_metric in copied_train_loss_meter_dict.items():
        copied_train_loss_meter_dict[loss_name].values = loss_metric.values[iters_per_epoch:]

    plot_metrics(range(1, iters_count - iters_per_epoch + 1), copied_train_loss_meter_dict, f"Training loss (2-{epochs_count} epochs)", get_plot_save_path("train_loss_trimmed", path))


def visualize_test_photo(model, path, color_space):
    test_photo_path = path
    test_paths = [test_photo_path]

    test_img = Image.open(test_photo_path)
    width, height = test_img.size

    test_dl = make_dataloaders(batch_size=1, paths=test_paths, split='val', color_space=color_space)
    data = next(iter(test_dl))
    visualize(model, data, color_space, False, width, height)