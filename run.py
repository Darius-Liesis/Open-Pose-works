import argparse
import logging
import sys
import time
import re, os, glob


from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

#Creates a list of files and sub directories
def getListOfFiles(dirName):
     
    #Names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    #Loops through all entries
    for entry in listOfFile:
        #Creates a full path
        fullPath = os.path.join(dirName, entry)
        #If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles   



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image_dir', type=str, default='./images/')     #User Input for the folder where all the images are located
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    #Code that reads image data existing in an entire directory
    image_array = getListOfFiles(args.image_dir)
    
    im = []
    #Loops through the ammount of images counter in the directory
    for elem in image_array:
        
        count = 0
        # estimate human poses from a single image !
        image = common.read_imgfile(elem, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % args.image)
            sys.exit(-1)
  
        #Write the image into the image_data directory
        image_name = os.path.basename(elem)
        cv2.imwrite(r'C:\Users\Administrator\Documents\GitHub\Open-Pose-works\images\image_data\%s' %image_name, image)
        
   # numbers = re.findall(r'\d+', args.image)
   # count = int(numbers[0])
    
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
        im.append(TfPoseEstimator.draw_humans(image_name, image, humans, imgcopy=False)) 
        image = im[0]     #Assigning the new value to the old default
        
    #cv2.imwrite(r'C:\Users\Administrator\Documents\GitHub\Open-Pose-works\images\image_results\frame%d_result.jpg' %count, image)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(1, 1, 1)
        a.set_title('Result')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

        # show network output
        #a = fig.add_subplot(2, 2, 2)
        #plt.imshow(bgimg, alpha=0.5)
        #tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        #plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        #plt.colorbar()

       #tmp2 = e.pafMat.transpose((2, 0, 1))
       #tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
       #tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

       #a = fig.add_subplot(2, 2, 3)
       #a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
       #plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
       #plt.colorbar()

       #a = fig.add_subplot(2, 2, 4)
       #a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
       #plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
       #plt.colorbar()
        #plt.show()
    except Exception as e:
        logger.warning('matplitlib error, %s' % e)
        cv2.imshow('result', image)
        cv2.waitKey()
