import numpy as np
import cv2
import matplotlib.pyplot as plt


# TODO: Implement program to detect curved lanes

# Function to stack videos for cleaner display output.
def stackVideos(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def toCanny(image):
    # Convert originalImage to grayscale
    originalToGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use canny() to identify edges of originalToGray
    return cv2.Canny(originalToGray, 50, 150)


def regionOfInterest(image):
    height = image.shape[0]

    # Use this for test1.mp4
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])

    # Use this for test2.mp4
    # triangle = np.array([[(200, height), (800, 350), (1200, height), ]], np.int32)

    # Apply this triangle on a black mask with the same dimensions as image
    mask = np.zeros_like(image)
    # Fill mask with white triangle
    cv2.fillPoly(mask, triangle, 255)
    # Combine Canny and Mask
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


def makeCoordinates(image, lineParameters):
    slope, intercept = lineParameters
    multiplier = 1/2
    # y1 is the height and y2 is 3/5 the height.
    # So our lines will both stop 3/5 the way up the image (for appearance).
    y1 = image.shape[0]
    y2 = int(y1 * multiplier)

    # x-values derives from y=mx+b
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def averageSlopeIntercept(image, lines):
    global leftLine, rightLine
    left = []  # Has negative slope
    right = []  # Has positive slope

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    # Average out slope and y-intercept in each of the left and right lists
    # Create lines from these averaged values
    if left:
        leftAverage = np.average(left, axis=0)
        leftLine = makeCoordinates(image, leftAverage)
    if right:
        rightAverage = np.average(right, axis=0)
        rightLine = makeCoordinates(image, rightAverage)

    return np.array([leftLine, rightLine])


# Now we've found the lines from our cropped image.
# We need to display these lines into colored image.
def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lineImage, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return lineImage


# load video
# test1 = "test1.mp4"
# cap = cv2.VideoCapture(test1)

test2 = "test2.mp4"
cap = cv2.VideoCapture(test2)

while cap.isOpened():
    _, frame = cap.read()
    # Convert colorImage to Canny
    cannyImage = toCanny(frame)

    # Find region of interest
    croppedImage = regionOfInterest(cannyImage)

    # Hough Transform
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # Refine appearance of lines
    averagedLines = averageSlopeIntercept(frame, lines)
    lineImage = displayLines(frame, averagedLines)

    # Combine lineImage and colorImage
    combinedImages = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)

    vidStack = stackVideos(0.5, ([cannyImage, croppedImage], [lineImage, combinedImages]))
    cv2.imshow("Video Comparisons", vidStack)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
