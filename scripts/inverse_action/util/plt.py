from matplotlib import pyplot as plt
import random
from PIL import Image
import numpy as np
import os

def show(image):
    plt.imshow(image)
    plt.show()

def show_torch(image):
    im = image.permute(1,2,0)
    plt.imshow(im)
    plt.show()

# convert pyplot image to rgb data in numpy array
def fig2img(fig):
    fname = f"/tmp/tmpimage{random.randint(100000,999999)}.png"
    fig.savefig(fname)
    out = np.array(Image.open(fname))
    os.system(f'rm {fname}')
    return out
