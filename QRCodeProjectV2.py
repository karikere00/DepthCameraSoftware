import cv2
import depthai
import numpy as np
from pyzbar.pyzbar import decode



#Includes authentication program w/ list of ID's to check
with open('myDataFile.text') as f:
    myDataList = f.read().splitlines()
print(myDataList)

# Create a pipeline object and add the OAK-D Lite camera node to the pipeline
pipeline = depthai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)
cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
cam.setFps(30)
cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("cam_out")
cam.preview.link(cam_xout.input)

# Create a barcode detection node and add it to the pipeline
detection = pipeline.createBarcodeDetection()
detection.setBarcodeType(depthai.BarcodeDetectionProperties.BarcodeType.QRCODE)
detection.setSearchArea(0.2, 0.2, 0.8, 0.8)
detection.setQuietZoneSize(4)
detection_xout = pipeline.createXLinkOut()
detection_xout.setStreamName("detection_out")
detection.out.link(detection_xout.input)


# Start the pipeline and retrieve the device queue
with depthai.Device(pipeline) as device:
    device_output_queues = [
        device.getOutputQueue(name="cam_out", maxSize=4, blocking=False),
        device.getOutputQueue(name="detection_out", maxSize=4, blocking=False),
    ]

    while True:
        # Get the output from the device queue
        in_queues = device.getQueueEvents()
        for queue_name, data in in_queues:
            if queue_name == "cam_out":
                img = data.getCvFrame()
                # Process the image data here
                for barcode in decode(img):
                    print(barcode.rect)
                    myData = barcode.data.decode('utf-8')
                    print(myData)

                    if myData in myDataList:
                        myOutput = 'Authorized'
                        myColor = (0, 255, 0) #Green
                    else:
                        myOutput = 'Not Authorized'
                        myColor = (0, 0, 255)  # Red

                    pts = np.array([barcode.polygon], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, myColor, 5)
                    pts2 = barcode.rect
                    cv2.putText(img,myOutput, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)

                cv2.imshow('Result', img)
                cv2.waitKey(1)
