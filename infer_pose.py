from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tensorflow as tf
from video_reader import VideoReader
import cv2


def infer_skater_poses(image, model='cmu', resize='0x0', resize_out_ratio=4.0):
    """
    :param image: CV Image read using IMREADCOLOR
    :param model: Inference model to use
    :param resize:
    :param resize_out_ratio:
    :return: a list of inferred humans and an image of them drawn on input.
    """
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # estimate skater poses from a single image !
    skaters = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    output = TfPoseEstimator.draw_humans(image, skaters, imgcopy=True)

    return skaters, output


def write_graph_to_tensorboard(model='cmu'):
    with tf.Session() as sess:
        model_filename = get_graph_path(model)
        with tf.gfile.GFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)

    LOGDIR = 'logs/'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)


vid_name = 'vid02'
game_watcher = VideoReader(vid_name)
skaters_drawn = None

write_graph_to_tensorboard()

while True:
    frame = game_watcher.read_frame(show_frame=False, color_format='bgr')
    if frame is None:
        break
    else:
        cv2.imshow('Input', frame)
        in_key = cv2.waitKey()
        if in_key == ord('q'):
            break
        elif in_key == ord('e'):
            if skaters_drawn is not None:
                cv2.imwrite('output/{}_{}.png'.format(vid_name, game_watcher.cur_frame-1), skaters_drawn)
        elif in_key == ord('w'):
            skaters, skaters_drawn = infer_skater_poses(frame)
            cv2.imshow('Skaters', skaters_drawn)


cv2.destroyAllWindows()
