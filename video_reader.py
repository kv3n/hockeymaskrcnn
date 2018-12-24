import cv2 as cv
from debug import Debug


class VideoReader:
    def __init__(self, filename):
        self.cur_frame = 0
        self.cap = cv.VideoCapture('data/' + filename + '.mp4')
        self.video_length = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        Debug().log('[Video Reader] Loading data/{}.mp4 of length {}'.format(filename, self.video_length))

    def __del__(self):
        self.reset_video()
        self.cap.release()
        cv.destroyAllWindows()
        Debug().log('[Video Reader] Closing Capture')

    def reset_video(self):
        self.cur_frame = 0

    def read_frame(self, show_frame=False, color_format='rgb'):
        ret, frame = self.cap.read()

        if not ret:
            self.reset_video()
            frame = None
        else:
            self.cur_frame += 1
            if show_frame:
                cv.imshow('Video Input', frame)
                in_key = cv.waitKey()
                if in_key == ord('q'):
                    self.reset_video()
                    frame = None

        if frame is not None:
            if color_format == 'rgb':
                conversion = cv.COLOR_BGR2RGB
            elif color_format == 'gray':
                conversion = cv.COLOR_BGR2GRAY
            elif color_format == 'hsv':
                conversion = cv.COLOR_BGR2HSV
            else:
                conversion = None

            if conversion is not None:
                frame = cv.cvtColor(frame, conversion)

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
