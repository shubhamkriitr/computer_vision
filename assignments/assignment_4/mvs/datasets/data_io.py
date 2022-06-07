import numpy as np
import re
import sys
from PIL import Image

GLOBAL_DATA_TYPE = np.float32

def read_cam_file(filename):
    # TODO
    intrinsics, extrinsics, depth_min, depth_max = None, None, None, None
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    
    lines = [l for l in lines if l.strip() != ""]

    if len(lines) != 10:
        raise AssertionError("The cam file should have exactly 10 non-empty lines")
    
    offset = 1
    # load extrinsic
    extrinsics = []
    for i in range(4):
        row_ = [float(x) for x in lines[offset + i].split()]
        extrinsics.append(row_)
    
    extrinsics = np.array(extrinsics, dtype=GLOBAL_DATA_TYPE)

    offset = 6

    intrinsics = []
    for i in range(3):
        row_ = [float(x) for x in lines[offset + i].split()]
        intrinsics.append(row_)
    
    intrinsics = np.array(intrinsics, dtype=GLOBAL_DATA_TYPE)

    depth_min, depth_max = [float(x) for x in lines[9].split()]

    return intrinsics, extrinsics, depth_min, depth_max

def read_img(filename):
    # TODO:A
    # FIXME - Should we use global stats to normalize ?

    # filename is actually path
    np_img = None

    im = Image.open(filename, "r")
    np_img = np.asarray(im)
    
    np_img = np_img/255  # image is in uint8
    return np_img.astype(GLOBAL_DATA_TYPE)

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


if __name__ == "__main__":
    im_path = "/home/shubham/Documents/Krishna/ETH/cv/computer_vision/assignments/assignment_4/res/rect_001_0_r5000.png"

    cam_paths = ["/home/shubham/Documents/Krishna/ETH/cv/computer_vision/assignments/assignment_4/res/0000000{}_cam.txt".format(i)
                for i in range(6)
                ]
    
    cam_data = []
    for cam_path in cam_paths:
        data = read_cam_file(cam_path)
        cam_data.append(data)

    im_arr = read_img(im_path)

    from matplotlib import pyplot as plt

    fig = plt.figure()
    plt.imshow(im_arr[:, :, 0])
    plt.show()

    print("Done")


