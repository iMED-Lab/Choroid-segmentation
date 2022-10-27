%%This code generate boundary image from segmentation ground truth. The
%%input is (H,W,n_images), 0~1 segmentation ground truth map

clear all;
close all;
clc

folder_path = '.'; %**********************Load data, Ex) PH2 dataset\DB\PH2
save_name = '/Documents/wwj/choroid/boundary.png';  % \boundary_PH2.mat%%
file_name = '/Documents/wwj/OCT_seg/label.png'; %**********************Load data, Ex) PH2 dataset\seg_PH2.mat

image_path = strcat(folder_path, file_name);
save_path = strcat(folder_path, save_name);
seg_PH2 = double(imread(image_path));  % load(image_path);%%
[h, w] = size(seg_PH2);  % %%
PH2_boundary = uint8(zeros(h, w)); %**********************Change according to the size of image (H,W,1)192,256,1
PH2_boundary = edge(seg_PH2, 'Canny');  % %%
% for i=1:size(seg_PH2, 3)
%     edge_image = edge(seg_PH2(:, :, i), 'Canny'); %********************** Generate edge map
% %     [edge_y,edge_x]
%     PH2_boundary(:, :, i) = uint8(edge_image); %**********************Generate edge map data (H,W,Number_of_image)
% end

imwrite(double(PH2_boundary), save_path);  % save(save_path,'PH2_boundary', '-v7.3'); %**********************Save
