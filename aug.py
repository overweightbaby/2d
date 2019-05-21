import numpy as np
from keras.preprocessing.image import flip_axis
import random
from scipy.ndimage.interpolation import rotate

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def random_rotation(x, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., theta=None):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        theta: Value to disable randomness or None.
    # Returns
        Rotated Numpy image tensor.
    """
    angle_list = [90,180,270]
    random.shuffle(angle_list)
    x= rotate(x, angle_list[0], axes=(0,1))
    return x



def random_channel_shift(x, intensity, channel_axis=2, known_intensity=None):
    x = np.rollaxis(x, channel_axis, 0)
    known_intensity = np.random.uniform(-intensity, intensity) if known_intensity is None else known_intensity
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + known_intensity, min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def flip_horizontal(x, col_axis=1):
    value = random.uniform(0, 1)
    if value < 0.5:
        a = 1
        x = x[:, ::-1,:]
    else:
        a = 0
    return x,a


def flip_vertical(x,  row_axis=0):
    value = random.uniform(0,1)
    if value < 0.5:
        a = 1
        x = x[::-1,:,:]
    else:
        a=0
        return x,a


def process(data_org,label):

    whole = []

    for i in range(2):

        r1 = random.uniform(0,1)
        if r1>0.5:
            angle_list = [90,180,270]
            random.shuffle(angle_list)
            data = random_rotation(x=data_org,rg=angle_list[0])

        data = flip_horizontal(data)
        data = flip_vertical(data)
        '''
        r3 = random.uniform(0,1)
        if r3 > 0.7 :
            intensity = random.randint(1,100)
            data = random_channel_shift(x=data,intensity=intensity)
        '''
        whole.append(data)

    whole = np.asarray(whole)
    print("whole",whole.shape)
    if label ==1 :
        labels = np.ones((whole.shape[0],1))
    else:
        labels = np.zeros((whole.shape[0],1))

    return whole,labels



def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

def brighter(image, percetage=1.2):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy
