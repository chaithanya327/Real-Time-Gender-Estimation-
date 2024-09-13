import cv2
import numpy as np
from my_gender import gender_test
import face_recognition

def get_face_info(im):
    boxes_face = face_recognition.face_locations(im)
    out = []
    if len(boxes_face)!=0:
        for box_face in boxes_face:
            box_face_fc = box_face
            x0,y1,x1,y0 = box_face
            box_face = np.array([y0,x0,y1,x1])
            face_features = {
                "gender":[],
                "bbx_frontal_face":box_face             
            } 

            face_image = im[x0:x1,y0:y1]


            face_features["gender"] = gender_test.gender(face_image)

            out.append(face_features)
    else:
        face_features = {
            "gender":[],
            "bbx_frontal_face":[]             
        }
        out.append(face_features)
    return out



def bounding_box(out,img):
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        if len(box) == 0:
            continue
        else:
            x0,y0,x1,y1 = box
            img = cv2.rectangle(img,
                            (x0,y0),
                            (x1,y1),
                            (0,255,0),2);
            thickness = 1
            fontSize = 0.5
            step = 13


            try:
                cv2.putText(img, "gender: " +data_face["gender"], (x0, y0-step-10*1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
 
    return img

