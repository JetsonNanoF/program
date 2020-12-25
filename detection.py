import jetson.inference
import jetson.utils
import cv2
import itertools

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video1")
display = jetson.utils.glDisplay()

try:
    while display.IsOpen:
        img, width, height = camera.CaptureRGBA()
        detections = net.Detect(img, width, height)
        display.RenderOnce(img, width, height)
        human_data = list()
        
        for detection in detections:
            if detection.ClassID == 1 and detecton.Confidence >= 0.60:
                human_data.append(detection)
         
        if len(human_data) >= 2:
            for a,b in list(itertools.combination(human_data,2)):
                if (abs(a.Center[0] - b.Center[0]) <= (a.Width /2 + b.Width / 2)):
                    print("Yes")
                else:
                    print("No")
        display.SetTitle("Objet Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        #display.close()
except KeyboardInterrupt:
    sys.exit()
