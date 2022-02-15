clc;
clear all;
close all;

dataset_0_dir = dir('group_0/airplane/*.jpg');
dataset_1_dir = dir('group_0/cat/*.jpg');
n_train = 450;
n_test = 50;
img_train = zeros(1024, n_train*2);
img_test = zeros(1024, n_test*2);
label_train = zeros(1, n_train*2);
label_test = zeros(1, n_test*2);

img_name = dataset_0_dir(4).name;
img = imread(['group_0/airplane/', img_name]);
img = rgb2gray(img);
eimg = edge(img,'sobel');
figure;
imshow(img);
figure;
imshow(eimg);
    
img_name = dataset_1_dir(473).name;
img = imread(['group_0/cat/', img_name]);
img = rgb2gray(img);
eimg = edge(img,'sobel');
figure;
imshow(img);
figure;
imshow(eimg);


