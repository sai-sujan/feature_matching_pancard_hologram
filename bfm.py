# # importing openCV library
import cv2

# function to read the images by taking there path
def read_image(path1,path2):
	read_img1 = cv2.imread(path1)
	read_img1 = cv2.resize(read_img1, (500, 500))
	read_img2 = cv2.imread(path2)
	read_img2 = cv2.resize(read_img2, (500, 500))

	return (read_img1,read_img2)

# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1,pic2):
	gray_img1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
	gray_img2 = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)
	return (gray_img1,gray_img2)

# function to detect the features by finding key points
# and descriptors from the image
def detector(image1,image2):
	# creating ORB detector
	detect = cv2.SIFT_create()
	#detect = cv2.AKAZE_create(threshold=0.0004)

	# finding key points and descriptors of both images using
	# detectAndCompute() function
	key_point1,descrip1 = detect.detectAndCompute(image1,None)
	key_point2,descrip2 = detect.detectAndCompute(image2,None)
	return (key_point1,descrip1,key_point2,descrip2)

# function to find best detected features using brute force
# matcher and match them according to there humming distance
def Knn_FeatureMatcher(des1,des2):
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	no_of_matches = flann.knnMatch(des1, des2, k=2)
	# brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
	# no_of_matches = brute_force.match(des1,des2)
	#
	# # finding the humming distance of the matches and sorting them
	# no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
	# return no_of_matches
	good = []
	#print(type(no_of_matches[0][0]))
	for m, n in no_of_matches:
		#print(m.distance,n.distance)
		if m.distance < 1 * n.distance:
			good.append(m)
			print(m.distance,n.distance)
	return good
# function displaying the output image with the feature matching
def display_output(pic1,kpt1,pic2,kpt2,best_match):

	# drawing the feature matches using drawMatches() function
	output_image = cv2.drawMatches(pic1,kpt1,pic2,kpt2,best_match,None,flags=2)
	imS = cv2.resize(output_image, (800, 800))
	cv2.imshow('Output image',imS)

# main function
if __name__ == '__main__':
	# giving the path of both of the images
	train_image_path = 'F:\API_username\holoback\Copy of IMG20220905124301.jpg'

	test_image_path = 'F:\API_username\holofront\IMG20220905123726.jpg'

	# reading the image from there paths
	img1, img2 = read_image(train_image_path,test_image_path)

	# converting the readed images into the gray scale images
	gray_pic1, gray_pic2 = convert_to_grayscale(img1,img2)

	# storing the finded key points and descriptors of both of the images
	key_pt1,descrip1,key_pt2,descrip2 = detector(gray_pic1,gray_pic2)

	# sorting the number of best matches obtained from brute force matcher
	number_of_matches = Knn_FeatureMatcher(descrip1,descrip2)
	tot_feature_matches = len(number_of_matches)

	# printing total number of feature matches found
	print(f'Total Number of Features matches found are {tot_feature_matches}')

	# after drawing the feature matches displaying the output image
	scale_percent = 220  # percent of original size
	width = int(gray_pic1.shape[1] * scale_percent / 100)
	height = int(gray_pic1.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized_gray_pic1 = cv2.resize(gray_pic1, dim, interpolation=cv2.INTER_AREA)
	resized_gray_pic2 = cv2.resize(gray_pic2, dim, interpolation=cv2.INTER_AREA)



	#display_output(resized_gray_pic1,key_pt1,resized_gray_pic2,key_pt2,number_of_matches)
	display_output(gray_pic1,key_pt1,gray_pic2,key_pt2,number_of_matches)

	cv2.waitKey()
	cv2.destroyAllWindows()

#
#
#






#
# # importing openCV library
# import cv2
#
# # function to read the images by taking there path
# def read_image(path1,path2):
# 	read_img1 = cv2.imread(path1)
# 	read_img2 = cv2.imread(path2)
# 	return (read_img1,read_img2)
#
# # function to convert images from RGB to gray scale
# def convert_to_grayscale(pic1,pic2):
# 	# gray_img1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
# 	# gray_img2 = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)
# 	gray_img1=pic1
# 	gray_img2 = pic2
# 	return (gray_img1,gray_img2)
#
# # function to detect the features by finding key points and descriptors from the image
# def detector(image1,image2):
# 	# creating ORB detector
# 	detect = cv2.ORB_create(nfeatures=5000)
#
# 	# finding key points and descriptors of both images using detectAndCompute() function
# 	key_point1,descrip1 = detect.detectAndCompute(image1,None)
# 	key_point2,descrip2 = detect.detectAndCompute(image2,None)
# 	return (key_point1,descrip1,key_point2,descrip2)
#
# # main function
# if __name__ == '__main__':
# # giving the path of both of the images
# 	first_image_path = 'F:\API_username\holoback\Copy of IMG20220905124301.jpg'
# 	second_image_path = 'F:\API_username\holofront\IMG20220905123726.jpg'
#
# 	# reading the image from there paths
# 	img1, img2 = read_image(first_image_path,second_image_path)
#
# 	# converting the readed images into the gray scale images
# 	gray_pic1, gray_pic2 = convert_to_grayscale(img1,img2)
#
# 	# storing the finded key points and descriptors of both of the images
# 	key_pt1,descrip1,key_pt2,descrip2 = detector(gray_pic1,gray_pic2)
#
# 	# showing the images with their key points finded by the detector
# 	cv2.imshow("Key points of Image 1",cv2.drawKeypoints(gray_pic1,key_pt1,None))
# 	cv2.imshow("Key points of Image 2",cv2.drawKeypoints(gray_pic2,key_pt2,None))
#
# 	# printing descriptors of both of the images
# 	print(f'Descriptors of Image 1 {descrip1}')
# 	print(f'Descriptors of Image 2 {descrip2}')
# 	print('------------------------------')
#
# 	# printing the Shape of the descriptors
# 	print(f'Shape of descriptor of first image {descrip1.shape}')
# 	print(f'Shape of descriptor of second image {descrip2.shape}')
#
# 	cv2.waitKey()
# 	cv2.destroyAllWindows()
#
