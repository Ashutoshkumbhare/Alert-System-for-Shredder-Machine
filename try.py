import os
import tensorflow as tf
from object_detection.utils import label_map_util
import cv2
import datetime
from imutils.video import VideoStream
import numpy as np

a=b=0


MODEL_NAME = 'faster_rcnn_resnet50'
PATH_TO_CKPT = 'C:\\Users\\Ashu\\Desktop\\Boxoffice\\models\\research\\faster_rcnn_resnet50' + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('C:\\Users\\Ashu\\Desktop\\Boxoffice\\models\\research\\object_detection\\data',
                              'mscoco_label_map.pbtxt')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_inference_graph():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess

def draw_box_on_image(num_bottle_detect, score_thresh, scores, boxes, classes,image_np):
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes
    print("inside draw box on image")

    global a,b
    hand_cnt=0
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    for i in range(num_bottle_detect):
        print("score = ",scores[0])
        print("score_thresh = ", score_thresh)
        print("Class = ", classes[0])

        if (scores[i] > score_thresh):
            if i == 0: color = color0
            else: color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            cv2.rectangle(image_np, p1, p2, color , 3, 1)


            # cv2.putText(image_np, 'hand '+str(i)+': '+id, (int(left), int(top)-5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            # cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
            #             (int(left),int(top)-20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            #
            # cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
            #             (int(im_width*0.65),int(im_height*0.9+30*i)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
            #
            # a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        # if hand_cnt==0 :
        #     b=0
        #     #print(" no hand")
        # else:
        #     b=1
        #     #print(" hand")

    return a,b


def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


detection_graph, sess = load_inference_graph()


if __name__ == '__main__':
    score_thresh = 0.30
    num_bottle_detect = 2
    num_frames = 0
    start_time = datetime.datetime.now()

    lst1 = []
    lst2 = []

    vs = VideoStream(1).start()
    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    while True:
        frame = vs.read()
        frame = np.array(frame)
        if im_height == None:
            im_height, im_width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run image through tensorflow graph
        boxes, scores, classes = detect_objects(
            frame, detection_graph, sess)


        # Draw bounding boxeses and text
        a, b = draw_box_on_image(
            num_bottle_detect,score_thresh, scores, boxes, classes, frame)

        cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            vs.stop()
            break
