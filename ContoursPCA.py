import cv2
import numpy as np
from math import *

def getRotateDegreeUsingPCA (src):
    # Convert image to grayscale
    print("1234")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Convert image to binary
    bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("1234")
    # cv2.imshow(bw)
    areas = []
    areas_angle = []
    # contours = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print("6789")
    #
    # for i, c in enumerate(contours):
    #     # Calculate the area of each contour
    #     area = cv2.contourArea(c)
    #     # Ignore contours that are too small or too large
    #     if area < 1e2 or 1e5 < area:
    #         continue
    #     areas.append(area)
    #     # Draw each contour only for visualisation purposes
    #     cv2.drawContours(src, contours, i, (0, 0, 255), 2)
    #     print("9999")
    #     # Find the orientation of each shape
    #     angle = getOrientation(c, src)
    #     areas_angle.append(angle)

    return areas_angle , bw

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # mean, eigenvectors, eigenvalues = cv.PCACompute(data_pts, mean, 2) #image, mean=None, maxComponents=10
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)  # 在PCA中心位置画一个圆圈
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)  # 绿色，较长轴
    drawAxis(img, cntr, p2, (255, 255, 0), 1)  # 黄色
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians #PCA第一维度的角度
    return angle
