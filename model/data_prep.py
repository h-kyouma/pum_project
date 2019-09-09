import os
import collections
import cv2
from keras.preprocessing.image import ImageDataGenerator
import math
import numpy as np
import pandas as pd
from random import randint


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def keras_augmentation():
    datagen = ImageDataGenerator(rotation_range=180,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 fill_mode='nearest',
                                 horizontal_flip=True,
                                 vertical_flip=True)

    if not os.path.exists('augmented'):
        os.makedirs('augmented')

    for dir_num in range(1, 5):
        if not os.path.exists('augmented/' + str(dir_num)):
            os.makedirs('augmented/' + str(dir_num))
        image_paths = os.listdir('photos/' + str(dir_num))
        for path in image_paths:
            print('\rAugmenting image: ' + str(dir_num) + '/' + path)
            image = cv2.imread('photos/' + str(dir_num) + '/' + path)
            if not os.path.isfile('augmented/' + str(dir_num) + '/' + path):
                cv2.imwrite('augmented/' + str(dir_num) + '/' + path, image)
            x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = x.reshape((1,) + x.shape)
            for x, val in zip(datagen.flow(x, save_to_dir='augmented/'+str(dir_num), save_prefix='aug', save_format='jpg'), range(5)):
                pass
    print('\nDONE\n')


def rotation_augmentation():
    if not os.path.exists('augmented'):
        os.makedirs('augmented')
    for dir_num in range(1, 5):
        if not os.path.exists('augmented/' + str(dir_num)):
            os.makedirs('augmented/' + str(dir_num))
        image_paths = os.listdir('photos/' + str(dir_num))
        for path in image_paths:
            print('\rAugmenting image: ' + str(dir_num) + '/' + path, end="     ")
            image = cv2.imread('photos/' + str(dir_num) + '/' + path)
            mask = np.zeros_like(image)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image_mask = cv2.inRange(image_hsv, np.array([80, 30, 30]), np.array([140, 255, 255]))
            _, contours, h = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            longest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [longest_contour], -1, (255, 255, 255), -1)
            (x, y, w, h) = cv2.boundingRect(longest_contour)
            image_roi = image[y:y + h, x:x + w]
            mask_roi = mask[y:y + h, x:x + w]
            image_extracted = np.zeros_like(image_roi)
            image_extracted[mask_roi == 255] = image_roi[mask_roi == 255]
            for angle in np.arange(0, 360, 60):
                randomizer = randint(-30, 31)
                angle = angle + randomizer
                rotated = rotate_bound(image_roi, angle)
                if randomizer % 3 == 0:
                    rotated = cv2.flip(rotated, 0)
                if randomizer % 5 == 0:
                    rotated = cv2.flip(rotated, 1)
                if randomizer % 7 == 0:
                    rotated = cv2.flip(rotated, -1)
                if not os.path.isfile('augmented/' + str(dir_num) + '/' + path.split('.')[0] + '_' + str(angle) + '.jpg'):
                    cv2.imwrite('augmented/' + str(dir_num) + '/' + path.split('.')[0] + '_' + str(angle) + '.jpg', rotated)
    print('\nDONE\n')


def convert_videos_to_frames():
    vids = [[subdir + '/' + f for f in os.listdir('videos/' + subdir)] for subdir in os.listdir('videos')]
    if not os.path.exists('photos'):
        os.makedirs('photos')
    for dir_num in range(1, 5):
        if not os.path.exists('photos/' + str(dir_num)):
            os.makedirs('photos/' + str(dir_num))
    for directory in vids:
        for vid in directory:
            print('\rConverting video: ' + vid)
            vidcap = cv2.VideoCapture('videos/' + vid)
            success, image = vidcap.read()
            count = 0
            while success:
                filename = 'photos/' + vid.split('/')[0] + '/{}_frame_{}.jpg'.format(vid.split('/')[1].split('.')[0], count)
                if not os.path.isfile(filename):
                    cv2.imwrite(filename, image)
                success, image = vidcap.read()
                if success is False:
                    print('Skipped a frame!', vid, count)
                count += 1
    print('\nDONE\n')


def mag(x):
    return math.sqrt(sum(i**2 for i in x))


def main(show=False, time=1000, convert_videos=False, augment_images=False, augmentation_method=None):
    if convert_videos is True:
        convert_videos_to_frames()

    main_directory = 'photos'
    if augment_images is True:
        main_directory = 'augmented'
        if augmentation_method == 'keras':
            keras_augmentation()
        elif augmentation_method == 'rotation':
            rotation_augmentation()
        else:
            print('Wrong augmentation method, possible options are: "keras", "rotation"')
            return

    contour_areas = []
    contour_lengths = []
    hull_areas = []
    hull_lengths = []
    rectangle_rotations = []
    moments_list = []
    points = {'left': [], 'right': [], 'top': [], 'bottom': []}
    labels = []

    paths = [os.listdir(main_directory + '/' + directory) for directory in os.listdir(main_directory)]
    for directory_num, directory in enumerate(paths):
        for filename in directory:
            print("\rProcessing image: " + str(directory_num + 1) + '/' + filename, end="     ")
            labels.append(directory_num)
            image = cv2.imread(main_directory + '/' + str(directory_num + 1) + '/' + filename)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            image_mask = cv2.inRange(image_hsv, np.array([80, 30, 30]), np.array([140, 255, 255]))
            if not os.path.exists('masks/' + str(directory_num + 1)):
                os.makedirs('masks/' + str(directory_num + 1))
            if not os.path.isfile('masks/' + str(directory_num + 1) + '/' + filename):
                cv2.imwrite('masks/' + str(directory_num + 1) + '/' + filename, image_mask)

            _, contours, h = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if show:
                image_contours = np.copy(image)
                cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 10, hierarchy=h, maxLevel=1)
                cv2.imshow(filename, cv2.resize(image_contours, (600, 800)))
                cv2.waitKey(time)
                cv2.destroyAllWindows()

            longest_contour = max(contours, key=cv2.contourArea)

            contour_area = cv2.contourArea(longest_contour)
            contour_areas.append(contour_area)
            contour_length = cv2.arcLength(longest_contour, True)
            contour_lengths.append(contour_length)

            convex_hull = cv2.convexHull(longest_contour)
            hull_area = cv2.contourArea(convex_hull)
            hull_areas.append(hull_area)
            hull_length = cv2.arcLength(convex_hull, True)
            hull_lengths.append(hull_length)

            # determine the most extreme points along the contour
            ext_left = tuple(longest_contour[longest_contour[:, :, 0].argmin()][0])
            ext_right = tuple(longest_contour[longest_contour[:, :, 0].argmax()][0])
            ext_top = tuple(longest_contour[longest_contour[:, :, 1].argmin()][0])
            ext_bot = tuple(longest_contour[longest_contour[:, :, 1].argmax()][0])

            points['left'].append(ext_left)
            points['right'].append(ext_right)
            points['top'].append(ext_top)
            points['bottom'].append(ext_bot)

            if show:
                image_hull = np.copy(image)
                cv2.drawContours(image_hull, [convex_hull], 0, (0, 0, 255), 10)
                cv2.circle(image_hull, ext_left, 10, (0, 0, 255), -1)
                cv2.circle(image_hull, ext_right, 10, (0, 255, 0), -1)
                cv2.circle(image_hull, ext_top, 10, (255, 0, 0), -1)
                cv2.circle(image_hull, ext_bot, 10, (255, 255, 0), -1)
                cv2.imshow(filename, cv2.resize(image_hull, (600, 800)))
                cv2.waitKey(time)
                cv2.destroyAllWindows()

            min_area_rect = cv2.minAreaRect(longest_contour)
            rectangle_rotation = min_area_rect[-1]
            rectangle_rotations.append(rectangle_rotation)
            box = cv2.boxPoints(min_area_rect)

            box = np.int0(box)
            if show:
                box_image = np.copy(image)
                cv2.drawContours(box_image, [box], 0, (255, 0, 0), 10)
                cv2.imshow(filename, cv2.resize(box_image, (600, 800)))
                cv2.waitKey(time)
                cv2.destroyAllWindows()

            if not os.path.exists('extracted/' + str(directory_num + 1)):
                os.makedirs('extracted/' + str(directory_num + 1))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(image_gray)
            cv2.drawContours(mask, [longest_contour], 0, 255, -1)
            image_extracted = np.zeros_like(image_gray)
            image_extracted[mask == 255] = image_gray[mask == 255]
            if not os.path.isfile('extracted/' + str(directory_num + 1) + '/' + filename):
                cv2.imwrite('extracted/' + str(directory_num + 1) + '/' + filename, image_extracted)
            if show:
                cv2.imshow(filename, cv2.resize(image_extracted, (600, 800)))
                cv2.waitKey(time)
                cv2.destroyAllWindows()

            moments = cv2.moments(image_extracted)
            moments_list.append(moments)

    moments_result = collections.defaultdict(list)
    for d in moments_list:
        for k, v in d.items():
            moments_result[k].append(v)

    df = pd.DataFrame(moments_result)
    df = df.assign(contour_areas=pd.Series(np.asarray(contour_areas)))
    df = df.assign(contour_lengths=pd.Series(np.asarray(contour_lengths)))
    df = df.assign(convex_hull_areas=pd.Series(np.asarray(hull_areas)))
    df = df.assign(convex_hull_lengths=pd.Series(np.asarray(hull_lengths)))
    df = df.assign(rectangle_rotations=pd.Series(np.asarray(rectangle_rotations)))
    df = df.assign(cnt_area_cnt_len_ratio=pd.Series(np.asarray(contour_areas) / np.asarray(contour_lengths)))
    df = df.assign(hull_area_hull_len_ratio=pd.Series(np.asarray(hull_areas) / np.asarray(hull_lengths)))
    df = df.assign(hull_area_cnt_area_ratio=pd.Series(np.asarray(hull_areas) / np.asarray(contour_areas)))
    df = df.assign(hull_len_cnt_len_ratio=pd.Series(np.asarray(hull_lengths) / np.asarray(contour_lengths)))
    df = df.assign(leftmost_point=pd.Series(np.asarray([e[0] for e in points['left']])))
    df = df.assign(leftmost_point_vect_mag=pd.Series(np.asarray([mag(point) for point in points['left']])))
    df = df.assign(rightmost_point=pd.Series(np.asarray([e[0] for e in points['right']])))
    df = df.assign(rightmost_point_vect_mag=pd.Series(np.asarray([mag(point) for point in points['right']])))
    df = df.assign(topmost_point=pd.Series(np.asarray([e[1] for e in points['top']])))
    df = df.assign(topmost_point_vect_mag=pd.Series(np.asarray([mag(point) for point in points['top']])))
    df = df.assign(bottommost_point=pd.Series(np.asarray([e[1] for e in points['bottom']])))
    df = df.assign(bottommost_point_vect_mag=pd.Series(np.asarray([mag(point) for point in points['bottom']])))
    df = df.assign(vertical_height=pd.Series(
        np.asarray([e[1] for e in points['bottom']]) - np.asarray([e[1] for e in points['top']])))
    df = df.assign(horizontal_height=pd.Series(
        np.asarray([e[0] for e in points['right']]) - np.asarray([e[0] for e in points['left']])))
    df = df.assign(vertical_to_horizontal_ratio=pd.Series(
        (np.asarray([e[1] for e in points['bottom']]) - np.asarray([e[1] for e in points['top']])) /
        (np.asarray([e[0] for e in points['right']]) - np.asarray([e[0] for e in points['left']]))))
    df = df.assign(label=pd.Series(np.asarray(labels)))
    df.to_csv('data.csv')

    print('\nDONE\n')


if __name__ == "__main__":
    main(show=True, time=0)
