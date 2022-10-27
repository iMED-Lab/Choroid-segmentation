% Create a logical image of a circle with specified
% diameter, center, and image size.
% First create the image.
% Generate ground truth key point map
clear all;
close all;
clc

load('./Documents/wwj/OCT_seg/point_PH2_6.mat');  % \DB\%%
point_endo_key = point_PH2_6;  % %%

% number_of_image = size(point_endo_key,3);%%
number_of_point = size(point_endo_key,1);

save_path = './PH2_boundary_key_point_map_GT.png';  % \DB\PH2_boundary_key_point_map_GT.mat%%
imageSizeX = 378; %*********************Set size of image192
imageSizeY = 379;  % 256%%
std = 8; %*****************************Circle diameter

[columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
attention_endo_key_std8 = double(zeros(imageSizeY, imageSizeX));
centerY=point_endo_key(:,2);
centerX=point_endo_key(:,1);
tmp_map=zeros(imageSizeY, imageSizeX);  % 256,320%%
for i=1:number_of_point
    circlePixels = (rowsInImage - centerY(i)).^2  + (columnsInImage - centerX(i)).^2 <= std.^2;
    tmp_map =tmp_map + double(circlePixels);
end
attention_endo_key_std8=tmp_map;
% for j=1:number_of_image
%     centerY=point_endo_key(:,2,j);
%     centerX=point_endo_key(:,1,j);
%     tmp_map=zeros(256,320);
%     for i=1:number_of_point
%         circlePixels = (rowsInImage - centerY(i)).^2  + (columnsInImage - centerX(i)).^2 <= std.^2;
%         tmp_map =tmp_map + double(circlePixels);
%     end
%     attention_endo_key_std8(:,:,j)=tmp_map;
% end
attention_endo_key_std8(attention_endo_key_std8>1)=1;
imwrite(double(attention_endo_key_std8), save_path);  % save(save_path,'attention_endo_key_std8', '-v7.3');
 
