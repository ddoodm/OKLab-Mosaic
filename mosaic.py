import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

source_img = Image.open('/Volumes/Phone SSD/DCIM/100APPLE/IMG_7014.HEIC')
sub_images_dirs = ['/Volumes/Phone SSD/DCIM/100APPLE']

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

tile_size = tuple((np.array(cell_size) * scale).astype(int))
cache_path = f'.tile_cache_{tile_size[0]}x{tile_size[1]}.pkl'

print('Loading sub images into memory ...')
cache = pickle.load(open(cache_path, 'rb')) if os.path.exists(cache_path) else {}
cache_dirty = False
sub_image_colors = []

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.heic', '.heif')
file_paths = [
    os.path.join(d, f)
    for d in sub_images_dirs
    for f in os.listdir(d)
    if not f.startswith('.') and f.lower().endswith(IMAGE_EXTENSIONS)
]

def load_tile(file_path):
    mtime = os.path.getmtime(file_path)
    entry = cache.get(file_path)
    if entry and entry['mtime'] == mtime:
        return file_path, mtime, entry['oklab'], entry['pixels'], False
    img = Image.open(file_path).convert('RGB')
    resized_img = resize_and_crop(img, tile_size)
    pixels = np.array(resized_img)
    oklab_color = rgb_to_oklab(average_color(resized_img))
    return file_path, mtime, oklab_color, pixels, True

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(load_tile, fp): fp for fp in file_paths}
    for future in tqdm(as_completed(futures), total=len(futures)):
        file_path, mtime, oklab_color, pixels, is_new = future.result()
        sub_image_colors.append((oklab_color, pixels))
        if is_new:
            cache[file_path] = {'mtime': mtime, 'oklab': oklab_color, 'pixels': pixels}
            cache_dirty = True

if cache_dirty:
    print('Saving tile cache ...')
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

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
    mosaic_image.paste(Image.fromarray(closest_sub_image), (int(x * scale), int(y * scale)))

mosaic_image.show()
