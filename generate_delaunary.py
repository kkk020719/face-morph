from imutils import face_utils
import numpy as np
import dlib
import cv2
import random
import time


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

#Function to draw circular point on image
def draw_point(img, p, color ) :
    cv2.circle( img, p, 1, color, -1, cv2.LINE_AA, 0 )


# Function to draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :

            cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, (int(pt2[0]),int(pt2[1])), (int(pt3[0]),int(pt3[1])), delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, (int(pt3[0]),int(pt3[1])), (int(pt1[0]),int(pt1[1])), delaunay_color, 1, cv2.LINE_AA, 0)

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

def delaunay_list(img1,img2,corresp):

    #img1 = cv2.imread(img1)
    size = img1.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    corresp = corresp.tolist()
    avg_points = []
    points_num = {}
    tri = []
    for (count,point) in enumerate(corresp):
        avg_points.append((int(point[0]), int(point[1])))
        points_num[(int(point[0]), int(point[1]))] = count


    for p in avg_points:
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList()

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            tri.append((points_num[pt1],points_num[pt2],points_num[pt3]))

    #return triangle list with points num
    return tri



"""""
if __name__ == '__main__':

    list1 = []
    #list2 = []
    image = cv2.imread("image_400.jpg");
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        img_orig = image.copy();
        size = image.shape
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect);
        #print(shape)
        # Insert points into subdiv
        for (count,p) in enumerate(shape):
            list1.append((int(p[0]),int(p[1])))
            subdiv.insert((int(p[0]),int(p[1])))
        list1.append((1, 1))
        list1.append((size[1] - 1, 1))
        list1.append(((size[1] - 1) // 2, 1))
        list1.append((1, size[0] - 1))
        list1.append((1, (size[0] - 1) // 2))
        list1.append(((size[1] - 1) // 2, size[0] - 1))
        list1.append((size[1] - 1, size[0] - 1))
        list1.append(((size[1] - 1), (size[0] - 1) // 2))

        subdiv.insert((1, 1))
        subdiv.insert((size[1] - 1, 1))
        subdiv.insert(((size[1] - 1) // 2, 1))
        subdiv.insert((1, size[0] - 1))
        subdiv.insert((1, (size[0] - 1) // 2))
        subdiv.insert(((size[1] - 1) // 2, size[0] - 1))
        subdiv.insert((size[1] - 1, size[0] - 1))
        subdiv.insert(((size[1] - 1), (size[0] - 1) // 2))
        # Draw delaunay triangles
        draw_delaunay(image, subdiv,  (255, 255, 255));

        # Draw points
        for p in shape:
            draw_point(image, (p[0],p[1]),  (0, 0, 255))

    cv2.imshow("Delaunay Triangle", image)

    cv2.waitKey(0)


"""""
