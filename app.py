from flask import Flask,render_template,Response, request
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import numpy as np
from PIL import Image
import gtts  
from playsound import playsound  
TEMPLATE_DIR = os.path.abspath('../templates')
STATIC_DIR = os.path.abspath('../static')



app = Flask(__name__)
global switch
switch =1


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])






def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=400)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            # for (box, pred) in zip(locs, preds):
            # 	# unpack the bounding box and predictions
            # 	(startX, startY, endX, endY) = box
            # 	(incorrectMask,mask, withoutMask) = pred

            # 	# determine the class label and color we'll use to draw
            # 	# the bounding box and text
            # 	if mask > incorrectMask and withoutMask:
            # 			label = "Mask" 
            # 	elif incorrectMask > mask and withoutMask:
            # 			label= "Incorrect Mask"
            # 	elif withoutMask > mask and incorrectMask:
            # 			label= "No mask"	
            # 	if label == "Mask":	
            # 			color = (0, 255, 0)  
            # 	elif label=="No mask":
            # 			color= (0, 0, 255)
            # 	else: 
            # 			color= (255, 165, 0)	
            count_nomask=0
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                if count_nomask==0:
                    prevlabel="No Mask"
                if label=="No Mask" and prevlabel=="No Mask":
                    count_nomask+=1
                    prevlabel=label

                
                if(count_nomask>300):
                    playsound("alert.mp3")

                    
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100-1.25)
                    
                # include the probability in the label
                # label = "{}: {:.2f}%".format(label, max(incorrectMask, mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()	
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')         
# @app.route('/style')
# def style():
#     return render_template('mystyle.css')
               
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

camera = cv2.VideoCapture(0)
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
    elif request.method=='GET':
        return render_template('index1.html')
    return render_template('index1.html')

# @app.route('/requests',methods=['POST','GET'])
# def tasks():
#     global switch,camera
#     if request.method == 'POST':
        
#         if  request.form.get('stop') == 'Stop/Start':
            
#             if(switch==1):
#                 switch=0
#                 camera.release()
#                 cv2.destroyAllWindows()
                
#             else:
#                 camera = cv2.VideoCapture(0)
#                 switch=1

#     elif request.method=='GET':
#         return render_template('index.html')
if __name__=="__main__":
    app.run(debug=True)