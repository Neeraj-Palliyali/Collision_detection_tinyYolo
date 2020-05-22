import cv2
import numpy as np
import time

windowName="Live video feed"
net = cv2.dnn.readNet("yolo-coco\yolov2-tiny.weights","yolo-coco\yolov2.cfg")
classes=[]
with open("yolo-coco/coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

colors=np.random.uniform(0,255,size=(len(classes),3))

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id=0

while True:
    _,frame = cap.read()

    H,W,C=frame.shape

    # detectinng objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs=net.forward(outputlayers)

    #show info on screen

    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.3:
                center_x= int(detection[0]*W)
                center_y=int(detection[1]*H)
                w=int(detection[2]*W)
                h=int(detection[3]*H)
                x=int(center_x - w/2)
                y=int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            label = str(classes[class_ids[i]])
            confidence=confidences[i]
            color=colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            if(label=="car"):
                print("car")
                mid_x=(boxes[i][3]+boxes[i][1])/2
                mid_y=(boxes[i][2]+boxes[i][0])/2
                apx_distance= round((boxes[i][3]-boxes[i][1])**4,1)
                cv2.putText(frame,"distance",(int(mid_x*800),int(mid_y*450)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                if apx_distance <= 0.5:
                    if mid_x<0.3 and mid_x<0.7:
                        cv2.putText(frame,"WARNING",(int(mid_x*800)-50,int(mid_y*450)-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)


    elapsed_time=time.time()-starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
 
    cv2.imshow("Image",frame)
    key=cv2.waitKey(1)

    if(key==27):
        break
cap.release()
cv2.destroyAllWindows()