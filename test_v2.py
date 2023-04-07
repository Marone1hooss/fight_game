import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
# from matplotlib import pyplot as plt
# from matplotlib.collections import LineCollection
# import matplotlib.patches as patches


# Some modules to display an animation using imageio.
import imageio
# from IPython.display import HTML, display


right_shoulder_ind = 6
right_elbow_ind = 8
right_wrist_ind = 10
right_hip_ind = 12


def cosinus(side_a, center, side_b) -> float:
    '''Return the cosinus of the angle
    formed by three points, centered on 
    the middle one. Using Al Khasi's formula
    '''
    hyp = (side_a[0] - side_b[0])**2 + (side_a[1] - side_b[1])**2
    adj_1 = ((center[0] - side_a[0])**2 + (center[1] - side_a[1])**2)
    adj_2 = ((center[0] - side_b[0])**2 + (center[1] - side_b[1])**2)    
    return (adj_1+adj_2-hyp)/(2*np.sqrt(adj_1*adj_2))


module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192


def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


def is_a_punch(keypoints):
    '''Determine whether or not a given 
    picture contain a human in a punching stance.
    '''

    '''EDIT : This function now takes the keys points
    instead of the pictures cuz I need the keypoints to be 
    avaliable outside of this function to compute the coordinate
    of the projectile
    '''

    shoulder = tuple(keypoints[0, 0, right_shoulder_ind, :])
    elbow = tuple(keypoints[0, 0, right_elbow_ind, :])
    wrist = tuple(keypoints[0, 0, right_wrist_ind, :])
    hip = tuple(keypoints[0, 0, right_hip_ind, :])

    cos_el_sh_hip = cosinus(elbow, shoulder, hip)
    cos_el_sh_wr = cosinus(elbow, shoulder, wrist)

    if (cos_el_sh_hip < 0.1 and cos_el_sh_hip > -0.1 and cos_el_sh_wr > 0.9):
        return True
    return False


def collide(rec_x, rec_y, rec_x2, rec_y2, sq_x, sq_y):
    '''trash collison detection function, just check if the lower right point
    of the square is in the target or not. 
    
                needs to be rewritten
    '''
    if sq_x < rec_x2 and sq_x > rec_x and sq_y < rec_y2 and sq_y > rec_y:
        return True
    return False



cap = cv2.VideoCapture(0)
cap.set(3,1000) # set Width
cap.set(4,680) # set Height


font = cv2.FONT_HERSHEY_SIMPLEX
frame = 0
curent_x = -1
curent_y = -1

while True:
    ret, img = cap.read()
    image = tf.convert_to_tensor(img)

    input_image = tf.expand_dims(image, axis=0)  #preparing the image for the movenet function
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size) 

    keypoints = movenet(input_image)  #take the key points so that I have the coordinate of every point avaible


    if collide(750, 100, 900, 500, curent_x, curent_y): 
        ''''check if the prejectile have hit yet the target.
        if so, destroy it, if not, make it advance
        '''
        cv2.rectangle(img, (750, 100), (900, 500), (0, 0, 255), -1)
        curent_x = -1
    elif curent_x != -1:
        cv2.rectangle(img, (curent_x-25, curent_y-25), (curent_x+25, curent_y+25), (255, 0, 0), -1)
        curent_x += 25


    '''create a point to keep track of where the wrist is
    in my computer the point is always off by a few cm, may be
    it's just the computing power that is low, or maybe I have
    used the wrong formula to calculate the coordinate, idk, 

    (However, the confidence of the points is verry low, maybe it's just that...)                  
    
                                needs debuging...
    '''
    cv2.circle(img, 
               (int(1000*keypoints[0, 0, right_wrist_ind, 1]), int(680*keypoints[0, 0, right_wrist_ind, 0])), 
               10, 
               (0, 255, 0), 
               -1)
    

    cv2.putText(img, 'Frame count :' + str(frame), (650, 50), font, 1, (255, 0, 0)) #Frame count because why not
    cv2.rectangle(img, (750, 100), (900, 500), (0, 0, 255), 1) #This is the target

    if is_a_punch(keypoints):
       cv2.putText(img, 'PUNCH !!', (50, 50), font, 1, (255, 0, 0))
       if curent_x == -1:
           '''compute the coordinate of the wrist (where the projectile should start)
           the 1000 and 680 are the width and height of the frame, I cast it to int cuz it's 
           the number of pixels...'''
           curent_x = int(1000*(keypoints[0, 0, right_wrist_ind, 1]))
           curent_y = int(680*(keypoints[0, 0, right_wrist_ind, 0]))
           print(curent_x, curent_y)
    
    cv2.imshow('video', img)

    frame+=1   #counting frames
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break  

cap.release()
cv2.destroyAllWindows()
