import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

# source_img = Image.open('/Volumes/Phone SSD/DCIM/100APPLE/IMG_7014.HEIC')
source_img = Image.open('IMG_7778.heic')
# sub_images_dirs = ['/Volumes/Phone SSD/DCIM/100APPLE']
sub_images_dirs = ['/Volumes/Phone SSD/DCIM/100APPLE', '/Volumes/Phone SSD/DCIM-Tian/100APPLE']

scale = 1.0
cell_size = (20, 20)

# Matching weights in OKLCh (cylindrical) space.
# Increase hue_weight to lock hue matching; chroma_weight to prefer saturated tiles;
# lower lightness_weight to let brightness vary more freely.
lightness_weight = 1.0
chroma_weight = 1.0
hue_weight = 1.0

# Small noise added to distances before picking the best tile.
# Creates dithering in regions where multiple tiles are near-equal matches.
# Raise to increase variety; lower to always pick the single best match.
dither_scale = 0.02


RGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

LMS_TO_OKLAB = np.array([
    [0.2104542553, +0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, +0.4505937099],
    [0.0259040371, +0.7827717662, -0.8086757660],
])


def linear_rgb_to_oklab(rgb_linear):
    lms = np.dot(rgb_linear, RGB_TO_LMS.T)
    return np.dot(np.cbrt(lms), LMS_TO_OKLAB.T)


def average_oklab(image):
    srgb = np.array(image) / 255.0
    # Linearize before averaging — averaging in gamma-encoded sRGB gives wrong results
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear_rgb_to_oklab(linear.mean(axis=(0, 1)))


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
        color_average = average_oklab(region)
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
    oklab_color = average_oklab(resized_img)
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

def oklab_to_lch(lab):
    """Convert OKLab (N, 3) or (3,) array to OKLCh [L, C, H_radians]."""
    L = lab[..., 0]
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    H = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack([L, C, H], axis=-1)


all_colors = np.array([color for color, _ in sub_image_colors])  # (n_tiles, 3)
all_pixels = [pixels for _, pixels in sub_image_colors]
all_lch = oklab_to_lch(all_colors)  # (n_tiles, 3): L, C, H
rng = np.random.default_rng()

print('Finding nearest fits and building image ...')
width, height = source_img.size
scaled_size = (int(width * scale), int(height * scale))
mosaic_image = Image.new('RGB', scaled_size)
for x, y, color_average in tqdm(color_averages):
    src_lch = oklab_to_lch(color_average)
    dL = lightness_weight * (all_lch[:, 0] - src_lch[0])
    dC = chroma_weight   * (all_lch[:, 1] - src_lch[1])
    # Circular hue difference, scaled by mean chroma to suppress noise for neutral colours
    raw_dH = all_lch[:, 2] - src_lch[2]
    raw_dH = np.arctan2(np.sin(raw_dH), np.cos(raw_dH))
    mean_C = (all_lch[:, 1] + src_lch[1]) / 2
    dH = hue_weight * mean_C * raw_dH
    noise = rng.standard_normal(len(all_lch)) * dither_scale
    distances = np.sqrt(dL ** 2 + dC ** 2 + dH ** 2) + noise
    idx = np.argmin(distances)
    mosaic_image.paste(Image.fromarray(all_pixels[idx]), (int(x * scale), int(y * scale)))

mosaic_image.show()
