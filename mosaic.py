import os
import numpy as np
from PIL import Image

source_img = Image.open('source.jpg')

region_size = (20, 20)

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


# Divide the source image into regions and compute the hash of each region
print('Building hashes ...')
region_hashes = []
for x in range(0, source_img.width, region_size[0]):
    for y in range(0, source_img.height, region_size[1]):
        region = source_img.crop((x, y, x + region_size[0], y + region_size[1]))
        region_hash = rgb_to_oklab(average_color(region))
        region_hashes.append((x, y, region_hash))

print('Loading sub images ...')
sub_images = []
sub_images_dir = 'floriasundays'
for filename in os.listdir(sub_images_dir):
    file_path = os.path.join(sub_images_dir, filename)
    if file_path.lower().endswith(('.jpg', '.jpeg')):
        img = Image.open(file_path)
        resized_img = img.resize(region_size)
        sub_images.append(resized_img)

print('Building sub hashes ...')
sub_image_hashes = [(rgb_to_oklab(average_color(img)), img) for img in sub_images]
hex_hashes = [str(item[0]) for item in sub_image_hashes]
print(hex_hashes)

print('Building mosaic image ...')
mosaic_image = Image.new('RGB', source_img.size)
for x, y, region_hash in region_hashes:
    # Find the sub image with the closest hash
    # closest_sub_image = min(sub_image_hashes, key=lambda item: region_hash - item[0])[1]
    closest_sub_image = min(sub_image_hashes, key=lambda item: np.linalg.norm(region_hash - item[0]))[1]
    # Place the closest sub image in the mosaic
    mosaic_image.paste(closest_sub_image, (x, y))

mosaic_image.show()
