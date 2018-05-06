clear
numFolds = 3; % should only be a divisor of 18 for balanced folding like 2,3,6,9(9 may be too much)

%Distributed data
dist_files_path = '/Users/sparshagarwal/OneDrive/Spring 2018/CS 567/Medical-Imaging/Project/distributed';
dist_files = dir(sprintf('%s/img*.ppm',dist_files_path));
dist_num_files = length(dist_files);
% for i=1:dist_num_files
%     dist_img(i,:,:,:) = double(imread(sprintf('%s/%s',dist_files_path, dist_files(i).name)));
% end
dist_labels = [zeros(1,18),ones(1,18)];


%%
% process distributed data
dist_feat1 = zeros(1,dist_num_files);
dist_feat2 = zeros(1,dist_num_files);
for j=1:dist_num_files
    dist_img = double(imread(sprintf('%s/%s',dist_files_path, dist_files(j).name)));
    one  = dist_img(:,:,1);
    two = dist_img(:,:,2);
    three = dist_img(:,:,3);
    % Take only the green channel
    img = two;
    % Make mask which will later be used to remove the edges of the eye
    mask = img>(0.1*max(max(img)));
    se = strel('disk',60,8);
    mask = imerode(mask,se);
    % Subtract median filtered image to remove intensity gradient
    img = img - medfilt2(img,[5 5]);
    img = img - min(min(img));
    img = (img./max(max(img)))*255;
    % Windowing function
    a = 0.61*max(max(img));
    b = 0.95*max(max(img));
    img(img<a) = 0;      
    img(img>b) = 255;
    img(img>=a & img<=b) = (255/(b-a))*img(img>=a & img<=b) - 255*a/(b-a);
    img(img<a) = 0;
    img(img>a) = 1;
    % Dilate spots so spots that are close by merge together
    se = strel('disk',20);
    img = imdilate(img,se);
    % Multiply image by mask to remove the outlines of the eye
    img = mask .* img;
    subplot(6,6,j);
    imagesc(img);
    colormap(gray);
%     img = two;
%     img = img - min(min(img));
%     img = (img./max(max(img)))*255;
%     % Windowing function
%     a = 0.75*max(max(img));
%     b = 0.9*max(max(img));
%     slope = 255/(b-a);
%     int = -1*a*255/(b-a);
%     rng = img>a & img<b;
%     img_wind = 0*img;
%     img_wind(rng) = img(rng)*slope + int;
%     img = img_wind;
%     img(img<=a) = 0;
%     img(img>a) = 1;
%     % Dilate spots so spots that are close by merge
%     modimg = zeros(size(img));
%     se = strel('disk',8);
%     modimg = imdilate(img,se);
%     img = modimg;
    CC = bwconncomp(img);
    dist_feat1(j) = CC.NumObjects; % Feature 1: Number of segments
    dist_feat2(j) = 100*length(find(img>0))/length(img(:));  % Feature 2: Percentage of image white after processing
end

%print feature vetors
% figure
% scatter([1:dist_num_files],dist_feat1);
% figure
% scatter([1:dist_num_files],dist_feat2);
figure
plot(dist_feat1(dist_labels==0), dist_feat2(dist_labels==0), 'r.', 'markersize', 10)
hold on
plot(dist_feat1(dist_labels==1), dist_feat2(dist_labels==1), 'g.', 'markersize', 10)
hold off