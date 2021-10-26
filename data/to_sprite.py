#!/usr/bin/env python

import sys
import numpy as np
from PIL import Image


if len(sys.argv) < 2:
    print("Usage: python to_sprite.py [file1.npy] ...")
    sys.exit()

for file in sys.argv[1:]:
    if not file.endswith('.npy'):
        continue
    data = np.load(file)[:, :, :, 3]
    nb_states = int(data.shape[0]**0.5)

    file_name = file.replace('.npy', '.jpg')

    sprite_image = Image.new(mode='L', size=(84*nb_states, 84*nb_states), color=0)

    for count, image in enumerate(data):
        x, y = divmod(count, nb_states)
        x_pos, y_pos = 84*x, 84*y
        sprite_image.paste(Image.fromarray(image), (x_pos, y_pos))

    sprite_image.convert("RGB").save(file_name, transparency=0)
