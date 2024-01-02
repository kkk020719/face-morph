import cv2
from imutils import face_utils
import dlib
import numpy as np
from generate_delaunary import delaunay_list, generate_face_correspondences, rect_contains
import imageio
from resize import crop_face

# Read points from text file
def readPoints(path):
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    #print(r)
    #print(t)

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    #print(str((1-mask).shape) + ' and ' + str(r[3]) + ' and ' +  str(r[2]))
    #print(str(imgRect.shape) + ' and ' + str(mask.shape))
    #print((img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask)).shape)
    #print(r[1])
    #print(r[1]+r[3])
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

def face_morph(img1,img2,points1,points2,tri_list,time=7,frames=24):

    num_animation = int(time*frames)
    image_lst = []

    for count in range(0,num_animation):
        #img1 = cv2.imread(filename1);
        #img2 = cv2.imread(filename2);

        alpha = count/(int(num_animation)-1)
        points = []

        for i in range(0,len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        img_morph = np.zeros(img1.shape, dtype=img1.dtype)

        for tri in tri_list:
            x = int(tri[0])
            y = int(tri[1])
            z = int(tri[2])

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            morphTriangle(img1, img2, img_morph, t1, t2, t, alpha)

        frame_rgb = cv2.cvtColor(np.uint8(img_morph), cv2.COLOR_BGR2RGB)
        image_lst.append(frame_rgb)
    imageio.mimsave('/Users/zhaotongke/Desktop/facemorph/morphing.gif', image_lst, fps=60)

if __name__ == '__main__':

    filename1 = 'alan.jpeg'
    filename2 = 'leo2.jpg'
    file1 = cv2.imread(filename1)
    file2 = cv2.imread(filename2)
    if file1.shape[:2] != file2.shape[:2]:
        file_list = crop_face(file1, file2)
    file1  = file_list[0]
    file2 = file_list[1]
    print(file1.shape)
    print(file2.shape)
    [correspondence_list, lists] = generate_face_correspondences(file1, file2)
    delaunay_tri = delaunay_list(file1, file2, correspondence_list)

    pts1 = lists[0]
    pts2 = lists[1]
    #print(lists)
    face_morph(file1, file2, pts1, pts2, delaunay_tri, time=7, frames=24)






