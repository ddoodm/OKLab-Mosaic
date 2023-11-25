import os
import numpy as np
from PIL import Image

source_img = Image.open('source.jpg')
sub_images_dir = 'floriasundays'

scale = 2.5

cell_size = (30, 30)


def rgb_to_oklab(rgb):
    # Normalize RGB values to the range [0, 1]
    rgb = rgb / 255.0

    # Linearize RGB values
    rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # Convert linear RGB to XYZ (using D65 illuminant)
    xyz_to_rgb_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb_linear, xyz_to_rgb_matrix.T)

    # Convert XYZ to LMS
    xyz_to_lms_matrix = np.array([
        [0.4002, 0.7075, -0.0808],
        [-0.2263, 1.1653, 0.0457],
        [0.0000, 0.0000, 0.9182]
    ])
    lms = np.dot(xyz, xyz_to_lms_matrix.T)

    # Convert LMS to Lab
    lms_to_lab_matrix = np.array([
        [1/np.sqrt(3), 0, 0],
        [0, 1/np.sqrt(6), 0],
        [0, 0, 1/np.sqrt(2)]
    ])
    lab = np.dot(np.sign(lms) * np.abs(lms) ** (1/3), lms_to_lab_matrix.T)

    return lab


def average_color(image):
    # Convert image to numpy array and calculate mean color
    data = np.array(image)
    mean_color = data.mean(axis=(0, 1))
    return mean_color


# Divide the source image into regions and compute the average color of each region in OKLab space
print('Finding OKLab coordinates of source image regions ...')
region_coords = []
for x in range(0, source_img.width, cell_size[0]):
    for y in range(0, source_img.height, cell_size[1]):
        region = source_img.crop((x, y, x + cell_size[0], y + cell_size[1]))
        region_coord = rgb_to_oklab(average_color(region))
        region_coords.append((x, y, region_coord))

print('Loading sub images into memory ...')
sub_images = []
for filename in os.listdir(sub_images_dir):
    file_path = os.path.join(sub_images_dir, filename)
    if file_path.lower().endswith(('.jpg', '.jpeg')):
        img = Image.open(file_path)
        resized_img = img.resize(tuple((np.array(cell_size) * scale).astype(int)))
        sub_images.append(resized_img)

print('Finding sub-image OKLab coordinates ...')
sub_image_coords = [(rgb_to_oklab(average_color(img)), img) for img in sub_images]

print('Finding nearest fits and building image ...')
width, height = source_img.size
scaled_size = (int(width * scale), int(height * scale))
mosaic_image = Image.new('RGB', scaled_size)
for x, y, region_coord in region_coords:
    # Find the sub image which is spatially closest to the region in the source image, in OKLab space.
    # Using the Euclidean distance between source region color and sub-image color
    closest_sub_image = min(sub_image_coords, key=lambda item: np.linalg.norm(region_coord - item[0]))[1]
    # Place the closest sub image in the mosaic
    mosaic_image.paste(closest_sub_image, (int(x * scale), int(y * scale)))

mosaic_image.show()
