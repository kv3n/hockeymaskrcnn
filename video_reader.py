import cv2 as cv
from debug import Debug


class VideoReader:
    def __init__(self, filename):
        self.cap = cv.VideoCapture('data/' + filename + '.mp4')
        Debug().log('[Video Reader] Loading data/{}.mp4'.format(filename))

    def __del__(self):
        self.cap.release()
        cv.destroyAllWindows()
        Debug().log('[Video Reader] Closing Capture')

    def read_frame(self, show_frame=False):
        ret, frame = self.cap.read()

        if not ret:
            frame = None
        elif show_frame:
            if show_frame:
                cv.imshow('Video Input', frame)
                in_key = cv.waitKey()
                if in_key == ord('q'):
                    frame = None

        return frame

"""
#####################################################################
## TEST
#####################################################################


vr = VideoReader('vid01')

while True:
    frame = vr.read_frame(show_frame=True)
    if frame is None:
        break
"""