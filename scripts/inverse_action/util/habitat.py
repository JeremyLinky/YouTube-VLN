import habitat
from habitat.utils.visualizations import maps
import numpy as np

def crop(img,padding=0):
    range_x = np.where(np.any(img, axis=1))[0]
    range_y = np.where(np.any(img, axis=0))[0]
    range_x = (
        max(range_x[0] - padding, 0),
        min(range_x[-1] + padding + 1, img.shape[0]),
    )
    range_y = (
        max(range_y[0] - padding, 0),
        min(range_y[-1] + padding + 1, img.shape[1]),
    )
    img = img[
        range_x[0] : range_x[1], range_y[0] : range_y[1]
    ]
    return img

def crop_to_range(img,rng):
    range_x, range_y = rng
    return img[ range_x[0] : range_x[1], range_y[0] : range_y[1] ]

def crop_range(img,padding=0):
    range_x = np.where(np.any(img, axis=1))[0]
    range_y = np.where(np.any(img, axis=0))[0]
    range_x = (
        max(range_x[0] - padding, 0),
        min(range_x[-1] + padding + 1, img.shape[0]),
    )
    range_y = (
        max(range_y[0] - padding, 0),
        min(range_y[-1] + padding + 1, img.shape[1]),
    )
    return range_x,range_y


def topdown_map(env,recolor = False):
    top_down_map = maps.get_topdown_map(env.sim, map_resolution=(5000, 5000))
    padding = int(np.ceil(top_down_map.shape[0] / 125))
    top_down_map = crop(top_down_map,padding)
    # background is 0
    # border is 1,
    # interior is 2
    # recolor
    if recolor:
        map_colors = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        top_down_map = map_colors[top_down_map]
    return top_down_map

def to_grid(pos,map_resolution):
    return maps.to_grid(pos[0], pos[2], maps.COORDINATE_MIN,
                        maps.COORDINATE_MAX, [map_resolution, map_resolution])

def from_grid(pos,map_resolution,height):
    x,y = maps.from_grid(pos[0], pos[1], maps.COORDINATE_MIN,
                        maps.COORDINATE_MAX, [map_resolution, map_resolution])
    return np.array([x,height,y])
