import cv2
import numpy as np

img1 = cv2.imread("A.jpg")
img2 = cv2.imread("B.jpg")

# image reshape
img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2))
img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2))

# gray
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# get key points
# describe key points
surf = cv2.xfeatures2d.SURF_create(800)
kp1, des1 = surf.detectAndCompute(gray1, None)
kp2, des2 = surf.detectAndCompute(gray2, None)

# draw key points
kpImg1 = cv2.drawKeypoints(img1, kp1, None)
kpImg2 = cv2.drawKeypoints(img2, kp2, None)

# FLANN 匹配器
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# 距离排序
matches = sorted(matches, key=lambda x: x[0].distance)

matchesMask = [[0, 0] for i in range(len(matches))]
# coff系数，决定有效关键点数量
coff = 0.25

# 匹配下限
MIN_MATCH_COUNT = 4
good_matches = []

for m, n in matches:
    if m.distance < coff * n.distance:
        good_matches.append(m)

if len(good_matches) > MIN_MATCH_COUNT:
    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = gray1.shape
    pts = np.float32([[0, 0], [0, h - 1],
                      [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    gray2 = cv2.polylines(gray2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    trans = cv2.perspectiveTransform(pts, M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
    found = cv2.warpPerspective(img2, perspectiveM, (w, h))

else:
    print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=2)

res = cv2.drawMatches(img1, kp1, img2, kp2,
                      good_matches, None, **draw_params)

cv2.imshow("Matches", res)
cv2.imshow("RAW", img2)
cv2.imshow("Result", found)
cv2.waitKey(0)
cv2.destroyAllWindows()
