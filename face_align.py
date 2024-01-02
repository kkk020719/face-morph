from imutils import face_utils
import numpy as np
import dlib
import cv2
import random
import time
from resize import crop_face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def generate_face_correspondences(img1, img2):

    list1 = []
    list2 = []
    j = 1
    img_list = [img1,img2]
    corresp = np.zeros((68, 2))

    for image in img_list:

        #image = cv2.imread(img);
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(image, 1)
        size = image.shape
        if (j == 1):
            currList = list1
        else:
            currList = list2

        for (i, rect) in enumerate(rects):
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)


            for (count,p) in enumerate(shape):
                currList.append((int(p[0]),int(p[1])))
                corresp[count][0] += int(p[0])
                corresp[count][1] += int(p[1])

            #adding 8 points of the background
            currList.append((1, 1))
            currList.append((size[1] - 1, 1))
            currList.append(((size[1] - 1) // 2, 1))
            currList.append((1, size[0] - 1))
            currList.append((1, (size[0] - 1) // 2))
            currList.append(((size[1] - 1) // 2, size[0] - 1))
            currList.append((size[1] - 1, size[0] - 1))
            currList.append(((size[1] - 1), (size[0] - 1) // 2))

            # Draw delaunay triangles
            #draw_delaunay(image, subdiv,  (255, 255, 255));


        j += 1

    corresp = corresp/2
    corresp = np.append(corresp, [[1, 1]], axis=0)
    corresp = np.append(corresp, [[size[1] - 1, 1]], axis=0)
    corresp = np.append(corresp, [[(size[1] - 1) // 2, 1]], axis=0)
    corresp = np.append(corresp, [[1, size[0] - 1]], axis=0)
    corresp = np.append(corresp, [[1, (size[0] - 1) // 2]], axis=0)
    corresp = np.append(corresp, [[(size[1] - 1) // 2, size[0] - 1]], axis=0)
    corresp = np.append(corresp, [[size[1] - 1, size[0] - 1]], axis=0)
    corresp = np.append(corresp, [[(size[1] - 1), (size[0] - 1) // 2]], axis=0)
    list_point = [list1,list2]

    return [corresp,list_point]

if __name__ == '__main__':
    filename1 = 'me2.jpeg'
    filename2 = 'leo2.jpg'
    file1 = cv2.imread(filename1)
    file2 = cv2.imread(filename2)
    if file1.shape[:2] != file2.shape[:2]:
        file_list = crop_face(file1, file2)
    file1 = file_list[0]
    file2 = file_list[1]
    [correspondence_list, lists] = generate_face_correspondences(file1, file2)
    print(lists)