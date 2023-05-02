import cv2
import tensorflow as tf
import keras
import numpy as np
from keras import backend as K

datadirectory = "C:\\Users\\Dell\\Downloads\\MaskDETECTION-master\\MaskDETECTION-master\\"
category_xml="haarcascade_frontalface_default.xml"
category_h5="mask_recog.h5"
face_cascade = cv2.CascadeClassifier(datadirectory+category_xml)
labels_dict={0:'no_musk',1:'musk'}
model = keras.models.load_model(datadirectory+category_h5)

video_capture=cv2.VideoCapture(0)

while video_capture.isOpened():
    _,img=video_capture.read()
    img=cv2.flip(img,1,1)
    mini=cv2.resize(img,(img.shape[1],img.shape[0]))
    blur=cv2.GaussianBlur(mini, (5,5), 0)
    dilated=cv2.dilate(blur, None,iterations=3)
    
    features=face_cascade.detectMultiScale(mini,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    coords = []
    for (x, y, w, h) in features:

        face=img[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_frame=cv2.GaussianBlur(face_frame, (5,5), 0)
        face_frame=cv2.dilate(face_frame, None,iterations=3)
        new_face=cv2.resize(face_frame,(224,224))
        normalize=new_face/255.0
        resize_face=np.reshape(normalize,(1,224,224,3))
        resize_face=np.vstack([resize_face])
        predict = model.predict(resize_face)
        for pred in predict:
          (mask, withoutMask) = pred

        if (predict[0][0] > predict[0][1] ):
            color, label = color_dict[1], "Mask"
        else:
            color, label = color_dict[0], "No_Mask"
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)    
        cv2.rectangle(img, (x, y), (x + w, y + h), color,thickness=2)
        cv2.putText(img, label, (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow("DetectFACE",img)
    if cv2.waitKey(1)==ord('q'):
        break;
video_capture.release()
cv2.destroyAllWindows()