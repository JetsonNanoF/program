import jetson.inference
import jetson.utils
import cv2
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "0")
display = jetson.utils.glDisplay()

try:
    i = 0
    while display.IsOpen:
        img, width, height = camera.CaptureRGBA()
        detections = net.Detect(img, width, height)
        display.RenderOnce(img, width, height)
        #print(detections)
        print(i)
        '''
        for detection in detections:
            if detection.ClassID == 1:
                print(detection)
        
        dist_matrix = [ [ minkowski([detecions[i].Center[0], detections[i].Center[1]], [detecions[j].Center[0], detections[j].Center[1] ], 1) for j in range(i, len(detections))] for i in range(len(detections)) ] 
        '''
        dist_matrix = [ minkowski([detecions[0].Center[0], detections[i].Center[1]], [detecions[j].Center[0], detections[j].Center[1] ], 1) ] 
        print()
        i += 1
        display.SetTitle("Objet Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        #display.close()
except KeyboardInterrupt:
    sys.exit()
