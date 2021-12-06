import glob
import json

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from PIL import Image


def solid_background_score(path_to_img, draw=False):
    img = Image.open(path_to_img)
    img.convert('RGB')
    img = np.array(img)
    if len(img.shape) != 3 or img.shape[2] != 3:
        return -1
    # colorReduce()
    div = 16
    img = img // div * div + div // 2
    offset = 0
    colors = {}
    record = -1
    saved_color = None
    while len(colors) <= 1:
        colors = {}
        try:
            border = img[0 + offset, :, :]
            border = np.concatenate((border, img[:, 0 + offset, :]))
            border = np.concatenate((border, img[:, img.shape[1] - 1 - offset, :]))
        except IndexError:
            print(path_to_img)
            return -1
        for pixel in border:
            pixel_as_tuple = tuple(pixel.tolist())
            if pixel_as_tuple in colors:
                colors[pixel_as_tuple] += 1
            else:
                colors[pixel_as_tuple] = 1
        offset += 1

    for color in colors:
        if colors[color] > record:
            record = colors[color]
            saved_color = color

    ratio = (np.sum(img == saved_color) / (img.shape[0] * img.shape[1])) / 3
    if draw:
        f, axes = plt.subplots(1, 2)
        hex_color = '#%02x%02x%02x' % saved_color
        axes[0].add_patch(Rectangle((0, 0), 10, 10,
                                    facecolor=hex_color,
                                    fill=True))
        axes[0].axis('off')
        axes[1].imshow(img)
        axes[1].axis('off')

        plt.show()
        print(ratio)

    return ratio


def save_processed(dictionary, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(dictionary))


def load_processed(filename):
    with open(filename) as f:
        dictionary = json.loads(f.read())
    return dictionary


def process_images(paths, filename):
    processed = {}
    for path in paths:
        score = solid_background_score(path)
        if score > 0:
            processed[path] = score
    save_processed(processed, filename)


def load_processed_images(filename, threshold=0.3):
    processed = load_processed(filename)
    return list(filter(lambda x: processed[x] < threshold, processed.keys()))


# path = "./data/part3"
# paths = glob.glob(path + "/*.jpg")
#
# process_images(paths, 'background_scores/filtered_part3')
# filtered = load_processed_images('test1')
