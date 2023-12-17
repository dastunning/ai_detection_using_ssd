from imutils.video import FPS
import numpy as np
import imutils
import cv2

use_gpu = True
live_video = False


confidence_level = 0.5
fps = FPS().start()
ret = True
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


print("[INFO] accessing video stream...")
if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture('person-bicycle-car-detection.mp4')

while ret:
    ret, frame = vs.read()
    if ret:
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)
        
        frame = imutils.resize(frame,height=400)
        cv2.imshow('Live detection',frame)

        if cv2.waitKey(1)==27:
            break

        fps.update()

fps.stop()

# print("start image detection")
# # Read the image
# image = cv2.imread('dog.jpg')
#
# # Resize the image
# image = imutils.resize(image, width=400)
#
# # Get image dimensions
# (h, w) = image.shape[:2]
#
# # Construct a blob from the image
# blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
#
# # Pass the blob through the network to get detections
# net.setInput(blob)
# detections = net.forward()
#
# # Loop over the detections
# for i in np.arange(0, detections.shape[2]):
#     # Extract confidence
#     confidence = detections[0, 0, i, 2]
#
#     # If confidence is greater than the confidence level
#     if confidence > confidence_level:
#         # Extract the index of the class label from the detections array
#         idx = int(detections[0, 0, i, 1])
#
#         # Compute the bounding box coordinates
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#
#         # Construct a label consisting of the class name and confidence
#         label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
#
#         # Draw a rectangle around the detected object
#         cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
#
#         # Determine the y-coordinate for the label
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#
#         # Draw the label on the image
#         cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)
#
# # Display the output image
# cv2.imshow('Image detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
