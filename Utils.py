import os
import cv2
import random
import numpy as np
import torch
from collections import namedtuple

from Data.prepro_siamese import crop_and_pad, get_context_size


### To reproduce the random data batch order and resuming.
def set_seed(seed=99):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        corner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        return np.array([x, y, w, h])


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return np.array([x1, y1, x2, y2])


def resize_padding(image, width, height, pad_mode=cv2.BORDER_CONSTANT, pad_value=0,
                   interpolation=None, pad_to='center'):
    assert pad_to in ['center', 'lefttop'], 'Invalid pad_to!!'
    o_size = image.shape[:2]
    d_size = (height, width)

    idx = np.argmax(o_size)
    ratio = float(d_size[idx]) / max(o_size)
    n_size = tuple([int(np.round(x * ratio)) for x in o_size])
    if n_size > d_size:
        idx = int(not idx)
        ratio = float(d_size[idx]) / min(o_size)
        n_size = tuple([int(np.round(x * ratio)) for x in o_size])

    target_h, target_w = n_size
    if interpolation is None:
        interpolation = cv2.INTER_AREA if o_size[0] * o_size[1] > target_h * target_w \
            else cv2.INTER_LINEAR
    image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    if pad_to == 'lefttop':
        pad_w = width - target_w
        pad_h = height - target_h
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, pad_mode, value=pad_value)
    else:
        pad_w = (width - target_w) // 2
        pad_h = (height - target_h) // 2
        image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, pad_mode, value=pad_value)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image, ratio, (pad_w, pad_h)


def get_template_image(image, bb, size, context_amount=0.5):
    if not isinstance(size, tuple):
        size = (size, size)

    cbb = corner2center(bb)
    t_sz = get_context_size(cbb, context_amount)
    image, _ = crop_and_pad(image, cbb[0], cbb[1], size, (t_sz, t_sz))
    return image


def select_target(fil):
    print('Controls: use `space` or `enter` to finish selection, use key `c` to cancel selection')
    print('Or If video use `c` to go next frame, use `ESC` to cancel')
    ext = os.path.splitext(fil)[-1]
    if ext in ['.jpg', '.JPEG', '.png']:
        image = cv2.imread(fil)
        bb = cv2.selectROI('Select target object', image)
        cv2.destroyAllWindows()
        if all(bb) == 0:
            return None, None
        bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
        return image, bb
    elif ext in ['.mp4', '.avi']:
        cap = cv2.VideoCapture(fil)
        res = None, None
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                bb = cv2.selectROI('Select target object', frame)
                k = cv2.waitKey(0)
                if k == 27:  # ESC
                    break
                if all(bb) == 0:
                    continue
                else:
                    bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
                    res = frame, bb
                    break

        cap.release()
        cv2.destroyAllWindows()
        return res
    else:
        print('Can not read file!')
        return None, None

