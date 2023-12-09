import cv2
import numpy as np
image1 = cv2.imread('88888.jpg')   # 示例图，给出内三角形的轮廓，用于比对
_, roi_thresh_inv = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(roi_thresh_inv, 50, 150)
inner_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

target_contour = max(inner_contours, key=cv2.contourArea)  # 使用判断所有轮廓的最大面积来筛选出需要的内三角形轮廓

image = cv2.imread('5.jpg')  # 需要识别的图

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

# 查找此图像中的轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

epsilon = 0.01 * cv2.arcLength(target_contour, True)
approx_corners_target = cv2.approxPolyDP(target_contour, epsilon, True)
pts2 = [corner[0] for corner in approx_corners_target]
pts2 = np.float32(pts2)



final_contours=[]
area_threshold = 100
maxarea_threshold = 2000
processed_contours = []

vertices = 3  # 顶点数

# 筛选出3个顶点的轮廓
for contour in contours:
    epsilon = 0.0224 * cv2.arcLength(contour, True)  # epsilon的值可以调整
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == vertices:  # 轮廓恰好有3个顶点
        processed_contours.append(approx)

for contour in processed_contours:
    if cv2.contourArea(contour) < area_threshold or cv2.contourArea(contour) > maxarea_threshold:
        continue
    match = cv2.matchShapes(target_contour, contour, 1, 0.0)
    if match < 0.8:  # 匹配阈值
        final_contours.append(contour)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
border_thickness =100    # 要增大的厚度
for contour in final_contours:
    # 计算轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(contour)

    # 扩展外接矩形的坐标和尺寸以包含外围的一圈图像
    x -= border_thickness
    y -= border_thickness
    w += 2 * border_thickness
    h += 2 * border_thickness

    # 确保坐标不小于0，以防止越界
    x = max(x, 0)
    y = max(y, 0)

    # 确保扩展后的矩形在图像边界内
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # 提取轮廓及其外围的一圈图像
    contour_with_border = image[y:y + h, x:x + w]

    # 再次通过cv2.approxPolyDP来近似轮廓，用来提取用于仿射变换的顶点
    epsilon = 0.0224 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    epsilon_target = 0.01 * cv2.arcLength(target_contour, True)
    approx_target = cv2.approxPolyDP(target_contour, epsilon_target, True)

    pts1 = np.float32([approx[0][0], approx[2][0], approx[1][0]])

    # 调整角点坐标到子图像的局部坐标系
    pts1 -= np.array([x, y])

    pts2 = np.float32([approx_target[0][0], approx_target[1][0], approx_target[2][0]])

    print(pts2)
    # 计算仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)

    # 应用仿射变换
    transformed_with_border = cv2.warpAffine(contour_with_border, M, (w, h))

    cv2.imshow('Transformed Image', transformed_with_border)
    cv2.waitKey(0)
