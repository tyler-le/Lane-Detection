import numpy as np
import cv2
import matplotlib.pyplot as plt


def toCanny(image):
    # Convert originalImage to grayscale
    originalToGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use canny() to identify edges of originalToGray
    return cv2.Canny(originalToGray, 50, 150)


def regionOfInterest(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    # Apply this triangle on a black mask with the same dimensions as image
    mask = np.zeros_like(image)
    # Fill mask with white triangle
    cv2.fillPoly(mask, triangle, 255)
    # Combine Canny and Mask
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


def makeCoordinates(image, lineParameters):
    slope, intercept = lineParameters

    # y1 is the height and y2 is 3/5 the height.
    # So our lines will both stop 3/5 the way up the image (solely for appearance).
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))

    # x-values derive from y=mx+b
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def averageSlopeIntercept(image, lines):
    left = []  # Has negative slope
    right = []  # Has positive slope
    if lines is not None:
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
    leftAverage = np.average(left, axis=0)
    rightAverage = np.average(right, axis=0)

    # Create lines from these averaged values
    leftLine = makeCoordinates(image, leftAverage)
    rightLine = makeCoordinates(image, rightAverage)

    return np.array([leftLine, rightLine])


# Now we've found the lines from our cropped image.
# We need to display these lines into colored image.
def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImage


# load image
colorImage = cv2.imread('test_image.jpg')
colorImageCopy = np.copy(colorImage)

cap = cv2.VideoCapture("test2.mp4")

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

    cv2.imshow("results", combinedImages)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
