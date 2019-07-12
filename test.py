from src.detector import detect_faces
from src.utils import show_bboxes
import numpy as np
from PIL import Image
import cv2
import math

def main():
    image = Image.open('images/test.jpg')
    opencv_image = cv2.imread("images/test.jpg")
    print("image size: ", image.size, ", opencv image size: ", opencv_image.shape)
    bounding_boxes, landmarks = detect_faces(image)
    print("bounding_boxes: ", bounding_boxes, ", landmarks: ", landmarks)
    # save face pic
    for bbox in bounding_boxes:
        cropped = opencv_image[int(bbox[1] - 20):int(bbox[3] + 20), int(bbox[0] - 20):int(bbox[2]) + 20]
        print("cropped size: ", cropped.shape)
        cv2.imshow("Face extract", cropped)

        # get eyes center xy
        lks = landmarks[0]
        center_x = (lks[0] + lks[1]) / 2  - bbox[0]
        center_y = (lks[5] + lks[6]) / 2  - bbox[1]

        # center_x = (lks[0] + lks[1]) / 2
        # center_y = (lks[5] + lks[6]) / 2
        eyesCenter = (center_x, center_y)

        # get eye angle
        dy = lks[6] - lks[5]
        dx = lks[1] - lks[0]
        angle = math.atan(dy/dx) / 3.14 * 180.0

        # get rototion matrix
        retval = cv2.getRotationMatrix2D(eyesCenter, angle, 1.0)

        print("angle: ", angle, ", retval: ", retval, ", eyesCenter: ", eyesCenter, ", cropped size: ", cropped.shape)
        # get warp affine
        #dst = cv2.warpAffine(opencv_image, retval, opencv_image.size)
        dst = cv2.warpAffine(cropped, retval, (cropped.shape[1], cropped.shape[0]))
        print("dst: ", type(dst), "len dst: ", len(dst), "dst shape: ", dst.shape)
        cv2.imshow("Face alignment", dst)


    image = show_bboxes(image, bounding_boxes, landmarks)
    image.show()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
