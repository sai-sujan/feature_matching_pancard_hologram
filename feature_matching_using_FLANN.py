
# # importing openCV library
import os
import time
import pickle
import cv2 as cv2



# function to read the images by taking there path
def read_image(path2):
    read_img2 = cv2.imread(path2)
    read_img2 = cv2.resize(read_img2, (500, 500))

    return (read_img2)


# function to convert images from RGB to gray scale
def convert_to_grayscale(pic2):
    gray_img2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    return (gray_img2)


# function to detect the features by finding key points
# and descriptors from the image
def detector(image2):
    # creating ORB detector
    detect = cv2.SIFT_create()

    key_point2, descrip2 = detect.detectAndCompute(image2, None)
    return (key_point2, descrip2)


# function to find best detected features using brute force
# matcher and match them according to there humming distance
def Flann_FeatureMatcher(des1, des2):
    #caculate the time taken to find the best matches
    start = time.time()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    no_of_matches = flann.knnMatch(des1, des2, k=2)
    end = time.time()
    #time taken in seconds
    print("Time taken to find the best matches: ", end - start)

    good = []
    for m, n in no_of_matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


# function displaying the output image with the feature matching
def display_output(pic1, kpt1, pic2, kpt2, best_match):
    # drawing the feature matches using drawMatches() function
    output_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)
    imS = cv2.resize(output_image, (800, 800))
    cv2.imshow('Output image', imS)


# main function
if __name__ == '__main__':
    folders = ['holoback','holofront','hololeft',"holoright"]
    matched_points=[]
    arr = []
    early_stop=False
    for i in folders:
        path1 = 'F:\\API_username\\'+i
        with open('descriptors.pkl', 'rb') as f:
            descriptors = pickle.load(f)

        with open('keypointers.pkl', 'rb') as f:
            key_pointers = pickle.load(f)

        final_list = []

        for arr in key_pointers:
            list1 = []
            for point in arr:
                temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3],
                                            octave=point[4], class_id=point[5])
                list1.append(temp_feature)
            final_list.append(list1)
        key_points_list = final_list

        key_points_for_image_number=0
        descriptor_for_image_number=0
        for key_point,descriptor in zip(key_points_list,descriptors):



            second_image_path = r'F:\API_username\images\holoimages\fake-holo.jpg'

            # reading the image from there paths
            img2 = read_image(second_image_path)

            # converting the readed images into the gray scale images
            gray_pic2 = convert_to_grayscale(img2)

            # storing the finded key points and descriptors of both of the images
            key_pt2, descrip2 = detector(gray_pic2)


            # sorting the number of best matches obtained from brute force matcher
            number_of_matches = Flann_FeatureMatcher(descriptor, descrip2)
            tot_feature_matches = len(number_of_matches)

            # printing total number of feature matches found
            print(f'Number of Features matches found are {tot_feature_matches}')
            matched_points.append(tot_feature_matches)
            if(tot_feature_matches>100):
                early_stop = True
                break
        if(early_stop==True):
            break
    print("Total number of feature matches are ",type(matched_points))
    flag=False
    for i in matched_points:
        if(i>100):
            flag=True




    if((sum(matched_points)>500 and flag) or early_stop):
        print("Real hologram")
    else:
        print("Fake hologram")









