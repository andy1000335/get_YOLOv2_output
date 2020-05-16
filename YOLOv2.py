import tensorflow as tf
import math
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tensorflow.python.platform import gfile

INPUTE_SIZE = 416
CLASS_NUM = 1
CELL_NUM = 13
ANCHOR_NUM = 5
THRESHOLD = 0.3
OVERLAP_THRESHOLD = 0.5

class BoundingBox():
    def __init__(self, x, y, w, h, confidence, classes):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.classes = classes

class BoundingBoxPosition():
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


def fill_image(img, inputSize):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inputSize

    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inputSize[1], inputSize[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(x):
    summ = 0
    params = np.zeros(len(x))
    for i in range(len(x)):
        params[i] = math.exp(x[i])
        summ += math.exp(x[i])
    if(np.isnan(summ) or summ<0):
        for i in range(len(x)):
            params[i] = 1 / len(x)
    else:
        for i in range(len(x)):
            params[i] = params[i] / summ
    return params

def argmax(params):
    maxIndex = 0
    for i in range(len(params)):
        if (params[maxIndex] < params[i]):
            maxIndex = i
    return maxIndex, params[maxIndex]

anchors = [0.672, 0.856, 0.406, 0.73866667, 0.898, 0.92168675, 0.268, 0.4024024, 0.59, 0.55466667]
def getAnchorBox(tensorFlowOutput, cellCol, cellRow, box, numClass, offset):
    x = (cellCol + sigmoid(tensorFlowOutput[offset])) * 32
    y = (cellRow + sigmoid(tensorFlowOutput[offset+1])) * 32
    w = math.exp(tensorFlowOutput[offset+2]) * anchors[2*box] * 32
    h = math.exp(tensorFlowOutput[offset+3]) * anchors[2*box + 1] * 32
    confidence = sigmoid(tensorFlowOutput[offset+4])
    classes = []
    for i in range(numClass):
        classes.append(tensorFlowOutput[i+offset+5])
    return BoundingBox(x, y, w, h, confidence, classes)

def getBestBoundingBox(bbox):
    label, param = argmax(softmax(bbox.classes))
    confidenceOfClass = param * bbox.confidence
    if (confidenceOfClass > THRESHOLD):
        tempLeft = bbox.x - bbox.w/2
        tempTop = bbox.y - bbox.h/2
        tempRight = tempLeft + bbox.w
        tempBottom = tempTop + bbox.h

        left = min(tempLeft, tempRight)
        right = max(tempLeft, tempRight)
        top = min(tempTop, tempBottom)
        bottom = max(tempTop, tempBottom)
        return [confidenceOfClass, BoundingBoxPosition(left, right, top, bottom), label]
    else:
        return None


def isOverlap(firstPosition, secondPosition):
    return ((firstPosition.left<secondPosition.right) and (firstPosition.right>secondPosition.left) 
            and (firstPosition.top<secondPosition.bottom) and (firstPosition.bottom>secondPosition.top))

def calculateIoU(firstPosition, secondPosition):
    if(isOverlap(firstPosition, secondPosition)):
        firstPositionArea = abs(firstPosition.right-firstPosition.left) * abs(firstPosition.bottom-firstPosition.top)
        secondPositionArea = abs(secondPosition.right-secondPosition.left) * abs(secondPosition.bottom-secondPosition.top)
        intersectionArea = (max(0, min(firstPosition.right, secondPosition.right)-max(firstPosition.left, secondPosition.left)) * 
                            max(0, min(firstPosition.bottom, secondPosition.bottom)-max(firstPosition.top, secondPosition.top)))

        unionArea = firstPositionArea + secondPositionArea - intersectionArea
        return intersectionArea / unionArea
    else:
        return 0

def NonMaximumSuppression(bboxes):
    bboxes = sorted(bboxes)
    positions = [position[1] for position in bboxes]    # get position object
    outputBoxes = []
    if(len(positions)>0):
        bestBox = positions.pop()
        outputBoxes.append(bestBox)
        for _ in range(len(positions)):
            secondaryBox = positions.pop()
            overlap = False
            for firstBox in outputBoxes:
                overlap = overlap or (calculateIoU(firstBox, secondaryBox) > OVERLAP_THRESHOLD)
            if(not overlap):
                outputBoxes.append(secondaryBox)
        return outputBoxes
    return outputBoxes



model = './yolov2_dog.pb'
originImage = Image.open('image/testImage.jpg')


originImage = np.array(originImage)
resizeImage = fill_image(originImage, [INPUTE_SIZE, INPUTE_SIZE])
image = resizeImage[np.newaxis, :, :, :]
image = (image-128)/128.0

with tf.Session() as sess:
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    x = sess.graph.get_tensor_by_name('input:0')
    y = sess.run(sess.graph.get_tensor_by_name('output:0'), {x: image})

positionList = []
for cellRow in range(CELL_NUM):
    for cellCol in range(CELL_NUM):
        offset = 0
        for box in range(ANCHOR_NUM):
            cell = y[0, cellRow, cellCol, :]
            boundingBox = getAnchorBox(cell, cellCol, cellRow, box, CLASS_NUM, offset)
            boundingBoxPosition = getBestBoundingBox(boundingBox)
            if boundingBoxPosition != None:
                positionList.append(boundingBoxPosition)
            offset = offset + 5 + CLASS_NUM            

recognation = NonMaximumSuppression(positionList)


plt.imshow(resizeImage)
currentAxis = plt.gca()
for i in range(len(recognation)):
    rectangle = patches.Rectangle((recognation[i].left, recognation[i].top), (recognation[i].right-recognation[i].left), 
                                  (recognation[i].bottom-recognation[i].top), 
                                  linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rectangle)
plt.show()
