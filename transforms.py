import numpy as np
import cv2

def order_points(points):
    '''Takes in 4 points and orders them in the following order:
    Top Left, Top Right, Bottom Right, Bottom Left

    :parameters:  
    points - 4 points in (x, y) format

    :returns: 
    numpy array containing points in order
    '''
    # Initialize empty rectangular shaped ndarray to store ordered points
    ordered_arr = np.zeros((4,2), dtype='float32')

    # Compute the sum and diffences between x and y of each point
    sum_coords = np.sum(points, axis=1)
    diff_coords = np.diff(points, axis=1)

    # Top left point has lowest sum
    ordered_arr[0] = points[np.argmin(sum_coords)]
    # Top right point has lowest difference
    ordered_arr[1] = points[np.argmin(diff_coords)]
    # Bottom right point has the greatest sum
    ordered_arr[2] = points[np.argmax(sum_coords)]
    # Bottom left point has the greatest difference
    ordered_arr[3] = points[np.argmax(diff_coords)]

    return ordered_arr

def distance(pt1, pt2):
    '''Computes the distance between two points

    :parameters:
    pt1 - first point
    pt2 - second point

    :returns:
    distance between the two points
    '''
    # Compute using distance formula sqrt((x1-x2)**2 + (y1-y2)**2)
    return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
    

def four_point_transform(image, points):
    '''Performs a four point perspective transform on an image

    :parameters:
    image - image to transform
    points - 4 points of the document

    :returns:
    image after 4 point transform
    '''
    # Ensure that points are in a consistent order so that operations are consistent
    points_ordered = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = points_ordered

    #### Compute desired dimensions for transformed image ####

    # Compute width of top edge
    top_width = distance(top_left, top_right)
    # Compute width of bottom edge
    bottom_width = distance(bottom_left, bottom_right)
    # Desired width is the greater of the two
    final_width = int(max(top_width, bottom_width))

    # Compute height of left edge
    left_height = distance(top_left, bottom_left)
    # Computer height of right edge
    right_height = distance(top_right, bottom_right)
    # Desired height is the greater of the two
    final_height = int(max(left_height, right_height))

    # Create an array with transformed coordinates
    points_transform = np.array([
        [0, 0],
        [final_width-1, 0],
        [final_width-1, final_height-1],
        [0, final_height-1]
    ], dtype='float32')

    # Compute transformation matrix
    M = cv2.getPerspectiveTransform(points_ordered, points_transform)
    image_transform = cv2.warpPerspective(image, M, (final_width, final_height))

    return image_transform