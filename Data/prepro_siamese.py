import cv2
import numpy as np


def round_up(value):
    return round(value + 1e-6 + 1000) - 1000


def get_context_size(center_box, context_amount):
    cx, cy, w, h = center_box  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    return s_z


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    if img_mean is None:
        img_mean = img.mean((0, 1)).astype(np.uint8)

    r, c, k = img.shape

    xmin = cx - (original_sz[0] - 1) / 2.
    xmax = xmin + original_sz[0] - 1
    ymin = cy - (original_sz[1] - 1) / 2.
    ymax = ymin + original_sz[1] - 1

    left = int(round(max(0., -xmin)))
    top = int(round(max(0., -ymin)))
    right = int(round(max(0., xmax - c + 1)))
    bottom = int(round(max(0., ymax - r + 1)))

    xmin = int(round(xmin + left))
    xmax = int(round(xmax + left))
    ymin = int(round(ymin + top))
    ymax = int(round(ymax + top))

    if any([top, bottom, left, right]):
        te_im = np.zeros((r + top + bottom, c + left + right, k), dtype=np.uint8) + img_mean
        te_im[top:top + r, left:left + c, :] = img

        im_patch = te_im[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch = img[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]

    if not model_sz == original_sz:
        # zzp: use cv to get a better speed
        im_patch = cv2.resize(im_patch.astype(np.uint8), model_sz)

    scale_x = float(model_sz[1]) / im_patch.shape[1]
    scale_y = float(model_sz[0]) / im_patch.shape[0]
    return im_patch, (scale_x, scale_y)


def get_instance_image(img, bbox, size_z, size_x, context_amount):
    img_mean = tuple(map(int, img.mean(axis=(0, 1))))

    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    scale_z = float(size_z) / s_z

    s_x = s_z * size_x / float(size_z)
    size_x, s_x = (size_x, size_x), (s_x, s_x)
    instance_img, (scale_x, scale_y) = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_y
    # point_1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
    # point_2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2
    # frame = cv2.rectangle(instance_img, (int(point_1[0]),int(point_1[1])), (int(point_2[0]),int(point_2[1])), (0, 255, 0), 2)
    # cv2.imwrite('1.jpg', frame)
    return instance_img, w_x, h_x, scale_x
