# -*- coding: utf-8 -*-
from __future__ import division
from PIL import ImageEnhance

import dlib
import numpy as np
import cv2
import sys
import os

predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
def getlandmarkdlib(im):
    # im=cv2.imread(imgpath)
    rects = detector(im, 1)
    points = [(p.x, p.y) for p in predictor(im, rects[0]).parts()]
    return points

def show68points(img,points,name):
    # img=cv2.imread(imgpath)
    for i in range(68):
        x = points[i][0]
        y = points[i][1]
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)
    cv2.imshow(name, img)
    #cv2.waitKey(0)


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

def whitening_face(img, faces, value=30):
    # value = 30

    imgw = np.zeros(img.shape, dtype='uint8')
    imgw = img.copy()
    midtones_add = np.zeros(256)

    for i in range(256):
        midtones_add[i] = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127))

    lookup = np.zeros(256, dtype="uint8")

    for i in range(256):
        red = i
        red += np.uint8(value * midtones_add[red])
        red = max(0, min(0xff, red))
        lookup[i] = np.uint8(red)

    # faces可全局变量

    if faces == ():
        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):
                imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                imgw[r, c, 2] = lookup[imgw[r, c, 2]]

    else:
        x = faces[0]
        y = faces[1]
        w = faces[2]
        h = faces[3]
        rows, cols, channals = img.shape
        x = max(x - (w * np.sqrt(2) - w) / 2, 0)
        y = max(y - (h * np.sqrt(2) - h) / 2, 0)
        w = w * np.sqrt(2)
        h = h * np.sqrt(2)
        rows = min(rows, y + h)
        cols = min(cols, x + w)
        for r in range(int(y), int(rows)):
            for c in range(int(x), int(cols)):
                imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                imgw[r, c, 2] = lookup[imgw[r, c, 2]]

        processWidth = int(max(min(rows - y, cols - 1) / 8, 2))
        for i in range(1, processWidth):
            alpha = (i - 1) / processWidth
            for r in range(int(y), int(rows)):
                imgw[r, int(x) + i - 1] = np.uint8(
                    imgw[r, int(x) + i - 1] * alpha + img[r, int(x) + i - 1] * (1 - alpha))
                imgw[r, int(cols) - i] = np.uint8(
                    imgw[r, int(cols) - i] * alpha + img[r, int(cols) - i] * (1 - alpha))
            for c in range(int(x) + processWidth, int(cols) - processWidth):
                imgw[int(y) + i - 1, c] = np.uint8(
                    imgw[int(y) + i - 1, c] * alpha + img[int(y) + i - 1, c] * (1 - alpha))
                imgw[int(rows) - i, c] = np.uint8(
                    imgw[int(rows) - i, c] * alpha + img[int(rows) - i, c] * (1 - alpha))

    return imgw

if __name__ == '__main__':

    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    # Read images
    # filename1 = './real/23060219910605594X.jpg'
    filename1 = './real/10.jpg'
    filename2 = './base/t0.jpg'
    # filename2 = './base/k0.jpg'

    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);

    sizeImg1 = img1.shape
    sizeImg2 = img2.shape

    img1=cv2.resize(img1,(sizeImg1[1]*1,sizeImg1[0]*1))
    img2=cv2.resize(img2,(int(sizeImg2[1]*1),int(sizeImg2[0]*1)))

    # Read array of corresponding points
    points1 = getlandmarkdlib(img1)
    points2 = getlandmarkdlib(img2)

    # show68points(img1,points1,"img1")
    #
    # show68points(img2,points2,"img2")
    img1Warped = np.copy(img2);
    # Find convex hull
    hull1 = []
    hull2 = []

    # hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    # hullIndex = [[8], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29],
    #               [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44],
    #               [45],[46], [47],[48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59]]

    hullIndex = [ [8], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29],
                 [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43],
                 [44], [45],[46], [47] ]

    # hullIndex_=[ [0],   [8], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29],
    #              [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45],
    #              [46], [47], [48], [60],[61], [62], [63], [64],[54],  [65], [66], [67] ]
    # hullIndex_=[ [0], [8], [16], [17],[18],[19],[20], [21],[22],[23],[24],[25], [26],[27],[28],[29],[30],
    #             [31],[32], [33], [34], [35],[36],[37], [38], [39],[40],[41], [42], [43], [44], [45],[46],
    #             [47],[48], [49], [50], [51], [52], [53],[54], [55], [56], [57], [58],[59],[60],[61],[62],[63],[64],[65],[66],[67] ]

    for i in range(0, len(hullIndex)):
        hull1.append(tuple(list(points1[int(hullIndex[i][0])])))
        hull2.append(tuple(list(points2[int(hullIndex[i][0])])))


    # generate extra points liunian_debug

    eye_distance1=0.5*(np.sqrt((points1[45][0]-points1[36][0])**2+(points1[45][1]-points1[36][1])**2)+np.sqrt((points1[42][0]-points1[39][0])**2+(points1[42][1]-points1[39][1])**2))
    eye_distance2=0.5*(np.sqrt((points2[45][0]-points2[36][0])**2+(points2[45][1]-points2[36][1])**2)+np.sqrt((points2[42][0]-points2[39][0])**2+(points1[42][1]-points1[39][1])**2))

    eye_height1=np.sqrt((points1[41][0]-points1[37][0])**2+(points1[41][1]-points1[37][1])**2)
    eye_height2=np.sqrt((points2[41][0]-points2[37][0])**2+(points2[41][1]-points2[37][1])**2)

    eye_l1= list(points1[36])
    eye_r1= list(points1[45])
    eye_t1= [0,0]
    eye_b1= [0,0]

    eye_l2= list(points2[36])
    eye_r2= list(points2[45])
    eye_t2= [0,0]
    eye_b2= [0,0]

    eye_alpha_lr=0.1
    eye_alpha_tb_t=0.2
    eye_alpha_tb_b=0.2
    # eye_alpha_tb_1=eye_alpha_tb_t*eye_distance1/eye_height1
    # eye_alpha_tb_2=eye_alpha_tb_t*eye_distance2/eye_height2

    # generate extra points liunian_debug
    eye_l1[0]=eye_l1[0]-int(eye_alpha_lr*(list(points1[45])[0]-list(points1[36])[0]))
    eye_r1[0]=eye_r1[0]+int(eye_alpha_lr*(list(points1[45])[0]-list(points1[36])[0]))
    eye_l1[1]=eye_l1[1]-int(eye_alpha_lr*(list(points1[45])[1]-list(points1[36])[1]))
    eye_r1[1]=eye_r1[1]+int(eye_alpha_lr*(list(points1[45])[1]-list(points1[36])[1]))

    eye_l2[0]=eye_l2[0]-int(eye_alpha_lr*(list(points2[45])[0]-list(points2[36])[0]))
    eye_r2[0]=eye_r2[0]+int(eye_alpha_lr*(list(points2[45])[0]-list(points2[36])[0]))
    eye_l2[1]=eye_l2[1]-int(eye_alpha_lr*(list(points2[45])[1]-list(points2[36])[1]))
    eye_r2[1]=eye_r2[1]+int(eye_alpha_lr*(list(points2[45])[1]-list(points2[36])[1]))

    d1=eye_alpha_tb_t*eye_distance1
    dd1=eye_alpha_tb_b*eye_distance1
    d2=eye_alpha_tb_t*eye_distance2
    dd2=eye_alpha_tb_b*eye_distance2

    theta1=np.arctan((eye_l1[1]-eye_r1[1])/(eye_l1[0]-eye_r1[0]))
    cos_theta1=np.cos(theta1)
    sin_theta1=np.sin(theta1)

    theta2=np.arctan((eye_l2[1]-eye_r2[1])/(eye_l2[0]-eye_r2[0]))
    cos_theta2=np.cos(theta2)
    sin_theta2=np.sin(theta2)

    ln_eye_l1=[eye_l1[0],eye_l1[1]]
    ln_eye_r1=[eye_r1[0],eye_r1[1]]
    ln_eye_t1=[eye_t1[0],eye_t1[1]]
    ln_eye_b1=[eye_b1[0],eye_b1[1]]

    ln_eye_l2=[eye_l2[0],eye_l2[1]]
    ln_eye_r2=[eye_r2[0],eye_r2[1]]
    ln_eye_t2=[eye_t2[0],eye_t2[1]]
    ln_eye_b2=[eye_b2[0],eye_b2[1]]

    ln_eye_l1[0]=int(eye_l1[0]+d1*sin_theta1)
    ln_eye_l1[1]=int(eye_l1[1]-d1*cos_theta1)
    ln_eye_r1[0]=int(eye_r1[0]+d1*sin_theta1)
    ln_eye_r1[1]=int(eye_r1[1]-d1*cos_theta1)
    ln_eye_t1[0]=int(eye_l1[0]-dd1*sin_theta1)
    ln_eye_t1[1]=int(eye_l1[1]+dd1*cos_theta1)
    ln_eye_b1[0]=int(eye_r1[0]-dd1*sin_theta1)
    ln_eye_b1[1]=int(eye_r1[1]+dd1*cos_theta1)



    ln_eye_l2[0]=int(eye_l2[0]+d2*sin_theta2)
    ln_eye_l2[1]=int(eye_l2[1]-d2*cos_theta2)
    ln_eye_r2[0]=int(eye_r2[0]+d2*sin_theta2)
    ln_eye_r2[1]=int(eye_r2[1]-d2*cos_theta2)
    ln_eye_t2[0]=int(eye_l2[0]-dd2*sin_theta2)
    ln_eye_t2[1]=int(eye_l2[1]+dd2*cos_theta2)
    ln_eye_b2[0]=int(eye_r2[0]-dd2*sin_theta2)
    ln_eye_b2[1]=int(eye_r2[1]+dd2*cos_theta2)



    distance_x1=list(points1[54])[0]-list(points1[48])[0]
    distance_y1=list(points1[54])[1]-list(points1[48])[1]
    mouse_distance1=np.sqrt(distance_x1**2+distance_y1**2)
    distance_x2=list(points2[54])[0]-list(points2[48])[0]
    distance_y2=list(points2[54])[1]-list(points2[48])[1]
    mouse_distance2=np.sqrt(distance_x2**2+distance_y2**2)

    height_x1=list(points1[57])[0]-list(points1[51])[0]
    height_y1=list(points1[57])[1]-list(points1[51])[1]
    mouse_height1=np.sqrt(height_x1**2+height_y1**2)
    height_x2=list(points2[57])[0]-list(points2[51])[0]
    height_y2=list(points2[57])[1]-list(points2[51])[1]
    mouse_height2=np.sqrt(height_x2**2+height_y2**2)



    # generate extra points liunian_debug
    mouse_l1= list(points1[48])
    mouse_r1= list(points1[54])
    mouse_t1= list(points1[33])
    mouse_b1= list(points1[57])

    mouse_l2= list(points2[48])
    mouse_r2= list(points2[54])
    mouse_t2= list(points2[33])
    mouse_b2= list(points2[57])

    #　extend points
    alpha_lr=0.1
    alpha_tb=0.1

    alpha_tb_1=alpha_tb*mouse_distance1/mouse_height1
    alpha_tb_2=alpha_tb*mouse_distance2/mouse_height2

    mouse_l1[0]=mouse_l1[0]-int(alpha_lr*(list(points1[54])[0]-list(points1[48])[0]))
    mouse_r1[0]=mouse_r1[0]+int(alpha_lr*(list(points1[54])[0]-list(points1[48])[0]))
    mouse_l1[1]=mouse_l1[1]-int(alpha_lr*(list(points1[54])[1]-list(points1[48])[1]))
    mouse_r1[1]=mouse_r1[1]+int(alpha_lr*(list(points1[54])[1]-list(points1[48])[1]))
    mouse_b1[0]= mouse_b1[0]+int(alpha_tb_1*(list(points1[57])[0]-list(points1[51])[0]))
    mouse_b1[1]= mouse_b1[1]+int(alpha_tb_1*(list(points1[57])[1]-list(points1[51])[1]))

    #

    mouse_l2[0]=mouse_l2[0]-int(alpha_lr*(list(points2[54])[0]-list(points2[48])[0]))
    mouse_r2[0]=mouse_r2[0]+int(alpha_lr*(list(points2[54])[0]-list(points2[48])[0]))
    mouse_l2[1]=mouse_l2[1]-int(alpha_lr*(list(points2[54])[1]-list(points2[48])[1]))
    mouse_r2[1]=mouse_r2[1]+int(alpha_lr*(list(points2[54])[1]-list(points2[48])[1]))
    mouse_b2[0]= mouse_b2[0]+int(alpha_tb_2*(list(points2[57])[0]-list(points2[51])[0]))
    mouse_b2[1]= mouse_b2[1]+int(alpha_tb_2*(list(points2[57])[1]-list(points2[51])[1]))


    ####　rotation
    a1=np.sqrt((mouse_l1[0]-mouse_t1[0])**2+(mouse_l1[1]-mouse_t1[1])**2)
    b1=np.sqrt((mouse_r1[0]-mouse_t1[0])**2+(mouse_r1[1]-mouse_t1[1])**2)
    c1=np.sqrt((mouse_l1[0]-mouse_r1[0])**2+(mouse_l1[1]-mouse_r1[1])**2)

    aa1=np.sqrt((mouse_l1[0]-mouse_b1[0])**2+(mouse_l1[1]-mouse_b1[1])**2)
    bb1=np.sqrt((mouse_r1[0]-mouse_b1[0])**2+(mouse_r1[1]-mouse_b1[1])**2)

    a2 = np.sqrt((mouse_l2[0] - mouse_t2[0]) ** 2 + (mouse_l2[1] - mouse_t2[1]) ** 2)
    b2 = np.sqrt((mouse_r2[0] - mouse_t2[0]) ** 2 + (mouse_r2[1] - mouse_t2[1]) ** 2)
    c2 = np.sqrt((mouse_l2[0] - mouse_r2[0]) ** 2 + (mouse_l2[1] - mouse_r2[1]) ** 2)

    aa2 = np.sqrt((mouse_l2[0] - mouse_b2[0]) ** 2 + (mouse_l2[1] - mouse_b2[1]) ** 2)
    bb2 = np.sqrt((mouse_r2[0] - mouse_b2[0]) ** 2 + (mouse_r2[1] - mouse_b2[1]) ** 2)

    s1=np.sqrt(0.5*(a1+b1+c1)*(0.5*(a1+b1+c1)-a1)*(0.5*(a1+b1+c1)-b1)*(0.5*(a1+b1+c1)-c1))
    ss1=np.sqrt(0.5*(aa1+bb1+c1)*(0.5*(aa1+bb1+c1)-aa1)*(0.5*(aa1+bb1+c1)-bb1)*(0.5*(aa1+bb1+c1)-c1))
    s2=np.sqrt(0.5*(a2+b2+c2)*(0.5*(a2+b2+c2)-a2)*(0.5*(a2+b2+c2)-b2)*(0.5*(a2+b2+c2)-c2))
    ss2=np.sqrt(0.5*(aa2+bb2+c2)*(0.5*(aa2+bb2+c2)-aa2)*(0.5*(aa2+bb2+c2)-bb2)*(0.5*(aa2+bb2+c2)-c2))

    d1=2*s1/(mouse_distance1+2*alpha_lr*mouse_distance1)
    dd1=2*ss1/(mouse_distance1+2*alpha_lr*mouse_distance1)
    d2=2*s2/(mouse_distance2+2*alpha_lr*mouse_distance2)
    dd2=2*ss2/(mouse_distance2+2*alpha_lr*mouse_distance2)

    theta1=np.arctan((mouse_l1[1]-mouse_r1[1])/(mouse_l1[0]-mouse_r1[0]))
    # tan_theta1=(mouse_l1[1]-mouse_r1[1])/(mouse_l1[0]-mouse_r1[0])
    cos_theta1=np.cos(theta1)
    sin_theta1=np.sin(theta1)

    theta2=np.arctan((mouse_l2[1]-mouse_r2[1])/(mouse_l2[0]-mouse_r2[0]))
    # tan_theta2=abs(mouse_l2[1]-mouse_r2[1])/abs(mouse_l2[0]-mouse_r2[0])
    cos_theta2=np.cos(theta2)
    sin_theta2=np.sin(theta2)

    ### box define
    ln_mouse_l1=[mouse_l1[0],mouse_l1[1]]
    ln_mouse_r1=[mouse_r1[0],mouse_r1[1]]
    ln_mouse_t1=[mouse_t1[0],mouse_t1[1]]
    ln_mouse_b1=[mouse_b1[0],mouse_b1[1]]
    ln_mouse_l2=[mouse_l2[0],mouse_l2[1]]
    ln_mouse_r2=[mouse_r2[0],mouse_r2[1]]
    ln_mouse_t2=[mouse_t2[0],mouse_t2[1]]
    ln_mouse_b2=[mouse_b2[0],mouse_b2[1]]

    ln_mouse_l1[0]=int(mouse_l1[0]+d1*sin_theta1)
    ln_mouse_l1[1]=int(mouse_l1[1]-d1*cos_theta1)
    ln_mouse_r1[0]=int(mouse_r1[0]+d1*sin_theta1)
    ln_mouse_r1[1]=int(mouse_r1[1]-d1*cos_theta1)
    ln_mouse_t1[0]=int(mouse_l1[0]-dd1*sin_theta1)
    ln_mouse_t1[1]=int(mouse_l1[1]+dd1*cos_theta1)
    ln_mouse_b1[0]=int(mouse_r1[0]-dd1*sin_theta1)
    ln_mouse_b1[1]=int(mouse_r1[1]+dd1*cos_theta1)

    ln_mouse_l2[0]=int(mouse_l2[0]+d2*sin_theta2)
    ln_mouse_l2[1]=int(mouse_l2[1]-d2*cos_theta2)
    ln_mouse_r2[0]=int(mouse_r2[0]+d2*sin_theta2)
    ln_mouse_r2[1]=int(mouse_r2[1]-d2*cos_theta2)
    ln_mouse_t2[0]=int(mouse_l2[0]-dd2*sin_theta2)
    ln_mouse_t2[1]=int(mouse_l2[1]+dd2*cos_theta2)
    ln_mouse_b2[0]=int(mouse_r2[0]-dd2*sin_theta2)
    ln_mouse_b2[1]=int(mouse_r2[1]+dd2*cos_theta2)


    hull1.append(tuple(ln_eye_l1))
    hull1.append(tuple(ln_eye_r1))
    hull1.append(tuple(ln_eye_t1))
    hull1.append(tuple(ln_eye_b1))
    hull1.append(tuple(ln_mouse_l1))
    hull1.append(tuple(ln_mouse_r1))
    hull1.append(tuple(ln_mouse_t1))
    hull1.append(tuple(ln_mouse_b1))

    hull2.append(tuple(ln_eye_l2))
    hull2.append(tuple(ln_eye_r2))
    hull2.append(tuple(ln_eye_t2))
    hull2.append(tuple(ln_eye_b2))
    hull2.append(tuple(ln_mouse_l2))
    hull2.append(tuple(ln_mouse_r2))
    hull2.append(tuple(ln_mouse_t2))
    hull2.append(tuple(ln_mouse_b2))

    # for j in range(0,len(hull1)):
    #     cv2.circle(img1,(hull1[j][0],hull1[j][1]),2,(0,255,0),-1)
    # cv2.imshow("img1_extra",img1)
    # cv2.waitKey(0)
    #
    # for j in range(0,len(hull2)):
    #     cv2.circle(img2,(hull2[j][0],hull2[j][1]),2,(0,255,0),-1)
    # cv2.imshow("img2_extra",img2)
    # cv2.waitKey(0)

    # Find delanauy traingulation for convex hull points

    rect = (0, 0, img2.shape[1], img2.shape[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    # dt=[(1,32,34) ,(1,2,32),(2,20,32),(2,20,21),(2,3,21),(3,4,8),(4,7,8),(3,4,21),(4,5,7),(4,21,22),(4,5,22),(5,6,7),(26,6,27),(6,11,26),(5,6,11),(7,8,28),(8,9,28),
    #     (7,27,28) ,(6,7,27),(28,29,30),(11,12,26),(9,10,28),(5,11,23),(28,29,33),(29,33,35),(21,24,25),(20,21,25),(5,22,23),(27,28,30),(26,27,31),(11,12,23),(21,22,24),
    #     (22,23,24),(12,13,23),(12,13,26),(13,19,31),(13,26,31),(13,23,24),(27,30,31),(13,14,15),(14,17,18),(14,18,19),(13,14,19),(14,16,17),(10,28,33),(20,32,34),(13,15,24),
    #     (14,15,16),(0,17,38),(0,17,39),(15,36,38),(15,16,38),(16,17,38),(17,18,39),(18,19,39),(19,37,39),(29,30,35),(25,20,34),(15,24,25),(19,30,31),(34,25,15),(35,19,30),
    #     (15,36,34),(19,35,37),(34,36,38),(35,37,39) ]

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img1, img1Warped, t1, t2)
        cv2.imshow('imgWarp', img1Warped)
        cv2.waitKey(0)

    # Calculate Mask
    hull8U = []
    # index = [0, 3, 7, 8, 12, 2, len(hull2) - 1, 1, len(hull2) - 2]
    # index = [len(hull2)-8, len(hull2)-7, len(hull2)-5, len(hull2)-3, len(hull2)-1, 0, len(hull2)-2, len(hull2)-4, len(hull2)-6]
    index = [len(hull2)-8, len(hull2)-7, len(hull2)-5, len(hull2)-1, 0, len(hull2)-2, len(hull2)-6]
    for i in range(0, len(index)):
        hull8U.append((hull2[index[i]][0], hull2[index[i]][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    # cv2.imshow(" mask--", np.uint8(mask))
    # cv2.waitKey(0)

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output1 = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    # detector = dlib.get_frontal_face_detector()
    # faces = detector(output1, 1)

    points_output = getlandmarkdlib(output1)

    # show68points(output1, points_output, "1")

    point_x = []
    point_y = []
    face = []
    for i in range(len(points_output)):
        point_x.append(points_output[i][0])
        point_y.append(points_output[i][1])
    x = min(point_x)
    w = max(point_x) - min(point_x)
    h = int(1.1 * max(point_y) - min(point_y))
    y = max(point_y) - h
    face.append(x)
    face.append(y)
    face.append(w)
    face.append(h)

    # cv2.rectangle(output1,(x,y),(x+w,y+h),(0,255,0),1)

    output2 = whitening_face(output1, face, value=50)

    # savename1 = '.src/jpg'
    # cv2.imwrite(savename1, output1)
    savename2 = './src1.jpg'
    cv2.imwrite(savename2, output2)

    cv2.imshow("Face Swapped", output1)
    cv2.waitKey(0)
    cv2.imshow("Face Swapped_", output2)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    #
    # detector = dlib.get_frontal_face_detector()
    # det=detector(output,1)
    # if len(det)!=0:
    #     height=abs(det[0].bottom()-det[0].top())
    #     if det[0].top()-int(0.2*height)>=0 and det[0].bottom()+int(0.2*height)<=output.shape[0]:
    #         bbox=output[det[0].top()-int(0.2*height):det[0].bottom()+int(0.1*height), det[0].left():det[0].right(), :]
    #     else:
    #         bbox = output[det[0].top():det[0].bottom(),det[0].left():det[0].right(), :]
    #
    #
    #
    # # bbox=cv2.imread("./real/000.jpg")
    # cv2.imshow("Face Swapped", bbox)
    # cv2.waitKey(0)
    #
    # # enh_bri = ImageEnhance.Brightness(bbox)
    # # brightness = 1.5
    # # image_brightened = enh_bri.enhance(brightness)
    #
    #
    # b,g,r = cv2.split(bbox)
    # b = cv2.equalizeHist(b)
    # g = cv2.equalizeHist(g)
    # r = cv2.equalizeHist(r)
    # merged = cv2.merge([b, g, r])
    # cv2.imshow("Face Swapped", merged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()







