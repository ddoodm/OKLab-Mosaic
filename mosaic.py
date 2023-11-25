import os

from PIL import Image
import imagehash

source_img = Image.open('source.jpg')

region_size = (20, 20)

# Divide the source image into regions and compute the hash of each region
print('Building hashes ...')
region_hashes = []
for x in range(0, source_img.width, region_size[0]):
    for y in range(0, source_img.height, region_size[1]):
        region = source_img.crop((x, y, x + region_size[0], y + region_size[1]))
        region_hash = imagehash.colorhash(region, 12)
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
sub_image_hashes = [(imagehash.colorhash(img, 12), img) for img in sub_images]
hex_hashes = [str(item[0]) for item in sub_image_hashes]
print(hex_hashes)

print('Building mosaic image ...')
mosaic_image = Image.new('RGB', source_img.size)
for x, y, region_hash in region_hashes:
    # Find the sub image with the closest hash
    closest_sub_image = min(sub_image_hashes, key=lambda item: region_hash - item[0])[1]
    # Place the closest sub image in the mosaic
    mosaic_image.paste(closest_sub_image, (x, y))

mosaic_image.show()
