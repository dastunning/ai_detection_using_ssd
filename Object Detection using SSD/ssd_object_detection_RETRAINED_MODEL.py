import cv2
from imutils.video import FPS
import numpy as np
import imutils
import tkinter as tk
from tkinter import filedialog

# Инициализация переменных
use_gpu = True
live_video = False
confidence_level = 0.5

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Use MobileNetSSD version 2
net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Функция для выбора режима (камера, видео или фото)
def select_mode():
    global live_video
    global vs
    mode_window.destroy()

    if var.get() == 0:  # Режим камеры
        live_video = True
        vs = cv2.VideoCapture(0)
    elif var.get() == 1:  # Режим видео
        live_video = True
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        vs = cv2.VideoCapture(file_path)
    else:  # Режим фото
        live_video = False
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            # Проверяем, является ли выбранный файл изображением
            img_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
            if file_path.lower().endswith(tuple(img_extensions)):
                vs = cv2.imread(file_path)
            else:
                print("[ERROR] Selected file is not an image.")
                return
        else:
            select_mode()  # Возврат в меню при отмене выбора файла
            return

    if not live_video:
        width = int(vs.shape[1])
        height = int(vs.shape[0])
        cv2.namedWindow('Image Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Detection', width, height)
    else:
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow('Video Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Detection', width, height)

# Создание окна выбора режима
mode_window = tk.Tk()
mode_window.title("Select Mode")
var = tk.IntVar()

camera_radio = tk.Radiobutton(mode_window, text="Camera", variable=var, value=0)
video_radio = tk.Radiobutton(mode_window, text="Video", variable=var, value=1)
photo_radio = tk.Radiobutton(mode_window, text="Photo", variable=var, value=2)
start_button = tk.Button(mode_window, text="Start Detection", command=select_mode)

camera_radio.pack(pady=10)
video_radio.pack(pady=10)
photo_radio.pack(pady=10)
start_button.pack(pady=20)

mode_window.mainloop()

fps = FPS().start()
video_opened = False

while True:
    if not live_video:
        frame = vs.copy()
    else:
        # Check if video has already been opened
        if not video_opened:
            video_opened = True
            ret, frame = vs.read()
            if not ret:
                break

        # Read next frame only for video
        ret, frame = vs.read()
        if not ret:
            break

    frame = imutils.resize(frame, width=800)

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

    if not live_video:
        cv2.imshow('Image Detection', frame)
    else:
        cv2.imshow('Video Detection', frame)

    # Выход из цикла при нажатии клавиши 'Esc'
    key = cv2.waitKey(1)
    if key == 27:
        break

    fps.update()

# Остановка и вывод статистики
fps.stop()
if fps.elapsed() > 0:
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
else:
    print("[INFO] Video is too short to calculate FPS.")

# Освобождение ресурсов и закрытие окна
if live_video:
    vs.release()
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()
