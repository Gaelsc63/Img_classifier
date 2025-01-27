This is an image classifier using my own dataset with fragrances, funkos, shoes, caps and shirts. It´s a very small dataset because it was for a school project and those were the requirements.
It works with KNN classifier. Reads the images in a gray scale with help of OpenCV function and gets a very complete feature extraction using Scikit-image functions.
After some tests the best results were obtained dividing the model in 20% test and 80% training with a value of 3 for n(nearest-neighbors) getting a 1.0 classification as a result.
