import numpy as np
import time
import cv2

labelsPath = r"path to coco.names file"
weightsPath_realtime = r"path to .weights file"
configPath_realtime = r"path to .cfg file"
weightsPath_video_and_photo = r"path to .weights file"
configPath_video_and_photo = r"path to .cfg file"
videoPath = r"path to video"
photoPath = r"path to photo"
class_obj = 'person'

gui_confidence = .5
gui_threshold = .3
prev_frame_time = 0
camera_number = 0
frame_id = 0
count_vid = 0
color2 = (0, 0, 0)
size_a = 640
size_b = 640

LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net_web = cv2.dnn.readNetFromDarknet(configPath_realtime, weightsPath_realtime)
ln_web = net_web.getLayerNames()
ln_web = [ln_web[i - 1] for i in net_web.getUnconnectedOutLayers()]

net_video_and_photo = cv2.dnn.readNetFromDarknet(configPath_video_and_photo, weightsPath_video_and_photo)
ln_video_and_photo = net_video_and_photo.getLayerNames()
ln_video_and_photo = [ln_video_and_photo[i - 1] for i in net_video_and_photo.getUnconnectedOutLayers()]

W, H = None, None
cap = cv2.VideoCapture(camera_number)
video = cv2.VideoCapture(videoPath)
frame = cv2.imdecode(np.fromfile(photoPath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
writer = None

def text0(frame, text, color, x, y):
    cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def text1_web(frame, count, color2, class_obj, fps):
    cv2.rectangle(frame, (0, 0), (400, 10), color2, 50)
    cv2.putText(frame, "No of " + class_obj + ":" + (str)(count), (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 0, 0), 2)
    cv2.putText(frame, "FPS:" + (str)(fps), (275, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 0, 0), 2)

def text1_photo_and_video(frame, count, color2, class_obj):
    cv2.rectangle(frame, (0, 0), (220, 17), color2, 20)
    cv2.putText(frame, "No of " + class_obj + ":" + (str)(count), (0, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 0, 0), 2)

def text2_web(frame, color2, class_obj, idxs, fps):
    if len(idxs) <= 0:
        cv2.rectangle(frame, (0, 0), (400, 10), color2, 50)
        cv2.putText(frame, "No of " + class_obj + ":0", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 0, 0), 2)
        cv2.putText(frame, "FPS:" + (str)(fps), (275, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 0, 0), 2)

def text2_photo_and_video(frame, count, color2, class_obj, idxs):
    if len(idxs) <= 0:
        cv2.rectangle(frame, (0, 0), (220, 17), color2, 20)
        cv2.putText(frame, "No of " + class_obj + ":" + (str)(count), (0, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 0, 0), 2)

def draw_boxes_web(idxs, classIDs, boxes, confidences, frame, color2, fps):
    count = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (LABELS[classIDs[i]]) == class_obj:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                            confidences[i])
                count = count + 1
                text0(frame, text, color, x, y)
            text1_web(frame, count, color2, class_obj, fps)
            if count == 0:
                text1_web(frame, count, color2, class_obj, fps)
    text2_web(frame, color2, class_obj, idxs, fps)

def draw_boxes_photo_and_video(idxs, classIDs, boxes, confidences, frame, color2):
    count = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (LABELS[classIDs[i]]) == class_obj:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                          confidences[i])
                count = count + 1
                text0(frame, text, color, x, y)
            text1_photo_and_video(frame, count, color2, class_obj)
            if count == 0:
               text1_photo_and_video(frame, count, color2, class_obj)
    text2_photo_and_video(frame, count, color2, class_obj, idxs)

def layer_outputs(layerOutputs, boxes, confidences, classIDs, W, H):
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > gui_confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

def realtime_from_webcam(frame_id, W, H, prev_frame_time):
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame_id += 1
        if not W or not H:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (size_a, size_b),
                                     swapRB=True, crop=False)
        net_web.setInput(blob)
        layerOutputs = net_web.forward(ln_web)
        boxes = []
        confidences = []
        classIDs = []

        layer_outputs(layerOutputs, boxes, confidences, classIDs, W, H)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

        draw_boxes_web(idxs, classIDs, boxes, confidences, frame, color2, fps)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def from_video(writer, count_vid, W, H):
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (size_a, size_b),
                                     swapRB=True, crop=False)
        net_video_and_photo.setInput(blob)
        layerOutputs = net_video_and_photo.forward(ln_video_and_photo)
        boxes = []
        confidences = []
        classIDs = []
        layer_outputs(layerOutputs, boxes, confidences, classIDs, W, H)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

        draw_boxes_photo_and_video(idxs, classIDs, boxes, confidences, frame, color2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter('output.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
            writer.write(frame)
            count_vid = count_vid + 1

    writer.release()
    video.release()

def from_photo(W, H):
    if not W or not H:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (size_a, size_b),
                                 swapRB=True, crop=False)
    net_video_and_photo.setInput(blob)
    layerOutputs = net_video_and_photo.forward(ln_video_and_photo)
    boxes = []
    confidences = []
    classIDs = []

    layer_outputs(layerOutputs, boxes, confidences, classIDs, W, H)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

    draw_boxes_photo_and_video(idxs, classIDs, boxes, confidences, frame, color2)

    cv2.imwrite("output.jpg", frame)
    cv2.imshow("output.jpg", frame)
    cv2.waitKey(0)

#realtime_from_webcam(frame_id, W, H, prev_frame_time)
#from_video(writer, count_vid, W, H)
#from_photo(W, H
