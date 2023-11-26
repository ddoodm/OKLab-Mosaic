import os
import numpy as np
from PIL import Image
from tqdm import tqdm

source_img = Image.open('source.jpg')
sub_images_dirs = ['floriasundays', 'sub_images']

scale = 1.0
reuse_penalty_factor = 0.0
cell_size = (20, 20)


def rgb_to_oklab(srgb):
    # Normalize sRGB values to the range [0, 1]
    srgb = srgb / 255.0

    # Linearize sRGB values
    rgb_linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

    # Convert linear RGB to LMS
    rgb_to_lms_matrix = np.array([
        [+0.4124564, +0.3575761, +0.1804375],
        [+0.2126729, +0.7151522, +0.0721750],
        [+0.0193339, +0.1191920, +0.9503041]
    ])
    lms = np.dot(rgb_linear, rgb_to_lms_matrix.T)

    # Convert LMS to OKLab
    lms_to_oklab_matrix = np.array([
        [+1/np.sqrt(3), 0, 0],
        [0, +1/np.sqrt(6), 0],
        [0, 0, +1/np.sqrt(2)]
    ])
    oklab = np.dot(np.cbrt(lms), lms_to_oklab_matrix.T)

    return oklab


def average_color(image):
    # Convert image to numpy array and calculate mean color
    data = np.array(image)
    mean_color = data.mean(axis=(0, 1))
    return mean_color


def resize_and_crop(img, output_size):
    # Scaling
    width, height = img.size
    aspect_ratio = width / height
    if aspect_ratio > 1:  # Width > Height
        new_width = int(output_size[0] * aspect_ratio)
        new_height = output_size[1]
    else:  # Width < Height
        new_width = output_size[0]
        new_height = int(output_size[1] / aspect_ratio)

    scaled_img = img.resize((new_width, new_height))

    # Cropping
    left = (scaled_img.width - output_size[0])/2
    top = (scaled_img.height - output_size[1])/2
    right = (scaled_img.width + output_size[0])/2
    bottom = (scaled_img.height + output_size[1])/2

    cropped_img = scaled_img.crop((left, top, right, bottom))

    return cropped_img


# Divide the source image into regions and compute the average color of each region in OKLab space
print('Finding OKLab coordinates of source image regions ...')
color_averages = []
for x in range(0, source_img.width, cell_size[0]):
    for y in range(0, source_img.height, cell_size[1]):
        region = source_img.crop((x, y, x + cell_size[0], y + cell_size[1]))
        color_average = rgb_to_oklab(average_color(region))
        color_averages.append((x, y, color_average))

print('Loading sub images into memory ...')
sub_images = []
for sub_images_dir in sub_images_dirs:  # Outer loop over directories
    for filename in tqdm(os.listdir(sub_images_dir)):  # Inner loop over files
        file_path = os.path.join(sub_images_dir, filename)
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            img = Image.open(file_path)
            resized_img = resize_and_crop(img, tuple((np.array(cell_size) * scale).astype(int)))
            sub_images.append(resized_img)

print('Finding sub-image OKLab coordinates ...')
sub_image_colors = [(rgb_to_oklab(average_color(img)), img) for img in sub_images]

# For tracking re-use
selection_counts = {tuple(color): 0 for color, _ in sub_image_colors}

print('Finding nearest fits and building image ...')
width, height = source_img.size
scaled_size = (int(width * scale), int(height * scale))
mosaic_image = Image.new('RGB', scaled_size)
for index, (x, y, color_average) in enumerate(tqdm(color_averages)):
    def distance_with_penalty(item):
        color, _ = item
        penalty = reuse_penalty_factor * selection_counts[tuple(color)]
        return np.linalg.norm(color_average - color) + penalty

    # Find the sub image which is spatially closest to the region in the source image, in OKLab space.
    # Using the Euclidean distance between source region color and sub-image color
    closest_color, closest_sub_image = min(sub_image_colors, key=distance_with_penalty)

    # Discourage this from being selected again
    selection_counts[tuple(closest_color)] += 1

    # Place the closest sub image in the mosaic
    mosaic_image.paste(closest_sub_image, (int(x * scale), int(y * scale)))

mosaic_image.show()
