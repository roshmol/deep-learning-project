

                                                #OBJECT DETECTION


import cv2
thres =0.45 # threshold to detect object



#img = cv2.imread("lena.png")
cap = cv2.VideoCapture(0)#The argument 0 specifies that the default camera should be used.

#These lines set various properties of the video capture object.
cap.set(3,1280)# sets the width of the video frame to 1280 pixels
cap.set(4,720)# sets the height to 720 pixels
cap.set(10,70)#sets the brightness level to 70.


classNames= []
classfile = 'coco.names'#These lines read the names of the objects that the model can detect from the coco.names
                        # file and store them in a list called classNames.
with open(classfile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath ="frozen_inference_graph.pb"#These lines specify the paths to the configuration file  and
                                    #  the pre-trained model weights that will be used for object detection.

net = cv2.dnn_DetectionModel(weightPath,configPath)#This line creates a dnn_DetectionModel object using the pre-trained model weights and
                                                # configuration file specified earlier.

#These lines set various input parameters for the object detection model.
net.setInputSize(320,320)#sets the input size of the model to 320x320 pixels,
net.setInputScale(1.0/ 127.5)#scales the input values to a range of [-1, 1],
net.setInputMean((127.5, 127.5, 127.5))# subtracts the mean RGB values of the dataset
net.setInputSwapRB(True)#swaps the Red and Blue color channels in the input.

while True:
    success,img = cap.read()#returns a boolean value indicating whether the frame was successfully read (success) and the image data (img)
                            #The cap.read() function reads a single frame from the video capture object
    classIds, confs, bbox =net.detect(img,confThreshold=thres)#The net.detect(img,confThreshold=thres)
                                                              # function performs object detection on the image data and
                                                              # returns the class IDs of the detected objects
    print(classIds,bbox)#bounding box coordinates (bbox). These values are printed to the console for debugging purposes.

    if len(classIds) != 0:

       for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)




    cv2.imshow("output",img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) == 27: # wait for ESC key to be pressed
        break

cv2.destroyAllWindows()
cap.release()
