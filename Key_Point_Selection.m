clear all;
close all;
clc

folder_path = '.'; %*************************************** Folder path\DB\PH2
file_name = '/Documents/wwj/OCT_seg//boundary.png'; %*************************************** Load boundary map (From boundary generate.m)\boundary_PH2.mat
save_name = '/Documents/wwj/choroid/point_PH2_6.mat'; %*************************************** Save file name\
seg_image = '/Documents/wwj/OCT_seg//label.png'; %*************************************** Load segmentation ground truth\seg_PH2.mat

image_path = strcat(folder_path,file_name);
seg_path = strcat(folder_path, seg_image);
save_path = strcat(folder_path, save_name);

PH2_boundary = imread(image_path) ./ 255;  % load(image_path);%%
seg_PH2 = imread(seg_path);  % load(seg_path);%%

[h, w] =size(seg_PH2);  % image_size =size(seg_PH2,3);%%

number_of_point = 6; %******************************* Set number of key point
point_PH2_6 = double(zeros(number_of_point, 2, 1)); %********************************** (y,x,n_image)

boundary_map = PH2_boundary;
seg_image = seg_PH2 ./ 255;
[boundary_y, boundary_x] = find(boundary_map==1); %****************************** Find boundary point candidate
number = size(boundary_y, 1);
best_dice = 0;
best_point_x = zeros(number_of_point, 1);
best_point_y = zeros(number_of_point, 1);
for n=1:40000
    p=randperm(number, number_of_point); %*************************Randomly selecte Boundary points
    rand_point_y = boundary_y(p);
    rand_point_x = boundary_x(p);
    
    k = boundary(rand_point_y, rand_point_x);
    if size(k) ~= number_of_point + 1
        continue;
    end
    bw = poly2mask(rand_point_x(k), rand_point_y(k), h, w); %*******************Connect selected key point and generate 'S'192,256
    % bw=uint8(bw);%%
    common = sum(bw(:) .* seg_image(:));
    dice = 2 * common/(sum(bw(:)) + sum(seg_image(:))); %**************************Calculate dice score (IOU) with ground truth
    if dice > best_dice
        best_dice = dice;
        best_point_x = rand_point_x(k);
        best_point_y = rand_point_y(k);
        best_bw = bw;
    end   
end
point_PH2_6 = cat(2, best_point_x(1:number_of_point), best_point_y(1:number_of_point)); %********************* Save key point
% for i=1:image_size
%     boundary_map = PH2_boundary(:,:,i);
%     seg_image = seg_PH2(:,:,i)/255;
%     [boundary_y, boundary_x] = find(boundary_map==1); %****************************** Find boundary point candidate
%     number = size(boundary_y,1);
%     best_dice = 0;
%     best_point_x = zeros(number_of_point,1);
%     best_point_y = zeros(number_of_point,1);
%     for n=1:40000
%         p=randperm(number,number_of_point); %*************************Randomly selecte Boundary points
%         rand_point_y = boundary_y(p);
%         rand_point_x = boundary_x(p);
%      
%         k=boundary(rand_point_y,rand_point_x);
%         if size(k)~=number_of_point+1
%             continue;
%         end
%         bw=poly2mask(rand_point_x(k),rand_point_y(k),192,256); %*******************Connect selected key point and generate 'S'
%         bw=uint8(bw);
%         common = sum(bw(:).*seg_image(:));
%         dice=2*common/(sum(bw(:))+sum(seg_image(:))); %**************************Calculate dice score (IOU) with ground truth
%         if dice>best_dice
%             best_dice = dice;
%             best_point_x = rand_point_x(k);
%             best_point_y = rand_point_y(k);
%             best_bw = bw;
%         end   
%     end
%     point_PH2_6(:,:,i)=cat(2,best_point_x(1:number_of_point),best_point_y(1:number_of_point)); %********************* Save key point
%     i
% end

save(save_path,'point_PH2_6', '-v7.3'); %************************
%% Visualize selected key point
figure;
imshow(boundary_map, []);
hold on;
scatter(best_point_x(1:number_of_point), best_point_y(1:number_of_point), 'g', 'filled');
% 
