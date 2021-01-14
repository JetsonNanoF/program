# python3 detection.py
# v4l2-ctl --list-devices
import jetson.inference
import jetson.utils
import cv2
import numpy as np
import itertools
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
font = jetson.utils.cudaFont()

camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
# display = jetson.utils.glDisplay()
try:
    i = 0
    # while display.IsOpen:
    while True:
        img, width, height = camera.CaptureRGBA(zeroCopy=1)
        detections = net.Detect(img, width, height)

        human_data = list()
        violate_num = 0
        for detection in detections:
            if detection.ClassID == 1 and detection.Confidence >=0.60:
                human_data.append(detection)

        if len(human_data) >= 2:
            for a,b in list(itertools.combinations(human_data,2)):
                if (abs(a.Center[0] - b.Center[0]) <= (a.Width /2 + b.Width/2) ):
		#if (abs(a.Center[1] - b.Center[1]) <= (a.Height /2 + b.Height /2) ): 
                    violate_num += 2
                else:
                    print("No!")

        #font.OverlayText(img,width,height,"{:4d}".format(violate_num),40,40,font.White,font.Gray40)

        img = jetson.utils.cudaToNumpy(img, width, height, 4)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2BGR)

        cv2.putText(img, "Contact:{:4d}".format(violate_num), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow("img_test", img)
        cv2.waitKey(1)

        '''
        display.SetTitle("Objet Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        except KeyboardInterrupt:
        sys.exit()
        display.close()
        '''

    cv2.destroyAllWindows()



except KeyboardInterrupt:
    sys.exit()
