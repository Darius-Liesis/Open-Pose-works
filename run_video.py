import argparse
import logging
import time
import os

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()
    
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    
    count = 0
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    while cap.isOpened():
        ret_val, image = cap.read()    #Reads the file from the video one frame at a time.
        
        #Writes the file into a JPEG, then calls the run.py file to process it.
        cv2.imwrite(r'C:/Users/Administrator/Documents/GitHub/Open-Pose-works/images/image_data/frame%d.jpg' %count, image)
        image_name = str("frame%d" %count) 
        #os.system('python run.py --model=mobilenet_thin --image=./images/video_images/frame%d.jpg' %count)
        
        #t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #elapsed = time.time() - t
        #print("humans has a ", type(humans), "type, and a value of ", humans)
        #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
        
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image_name, image, humans, imgcopy=False)
        #cv2.imwrite(r'C:\Users\Administrator\Documents\GitHub\Open-Pose-works\images\image_results\frame%d_result.jpg' %count, image)
    
        count += 1
        
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
