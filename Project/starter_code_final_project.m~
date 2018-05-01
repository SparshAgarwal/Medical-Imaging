path = '/Users/sparshagarwal/OneDrive/Spring 2018/CS 567/Medical-Imaging/Project/distributed';

% This will generate a list of all the files
files = dir(sprintf('%s/img*.ppm',path));
num_files = length(files);
%% 

filt_size = [3,5, 11, 21];
figure
for i=1:length(filt_size)
    filt_dat = medfilt2(epith, [filt_size(i), filt_size(i)]);
    subplot(4, 2, 2*i-1)
    hist(filt_dat(:), 50)
    subplot(4, 2, 2*i)
    imagesc(filt_dat); colormap('gray')
end
%% 

figure
for i=1:num_files
    img = double(imread(sprintf('%s/%s',path, files(i).name)));
    img = img(:,:,2);
    % figure
    % hist(img(:), 50)
    m = [-1, -1, -1; -1, 9, 1; -1, -1, -1];
    filt_dat_unsharp_mask = imfilter(img,m);
    filt_dat_median = medfilt2(img, [11 11]);
    filt_dat = img - filt_dat_median;
    filt_dat(:,:) = filt_dat(:,:)+100;
    filt_dat(filt_dat<90 & filt_dat>110) = 255;
    filt_dat(filt_dat>=90 & filt_dat<=110) = 0;
    % figure
    % imagesc(img)
    % figure
    % imagesc(filt_dat_unsharp_mask)
    % figure
    % imagesc(filt_dat_median)
    subplot(6,6,i)
%     figure
    imagesc(filt_dat)
    colormap(gray)
%     hist(filt_dat(:), 200)
end
%% 


% Not terribly useful (since we already did this in homework), but a quick
% check of each color channel
% figure
% for i=1:num_files
%     img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
%     subplot(6,6,i)
%     colormap(gray)
%     imagesc(img_loop(:,:,1))
% end

% Make a figure of all the green channels
% figure
% for i=1:num_files
%     img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
%     subplot(6,6,i)
%     colormap(gray)
%     imagesc(img_loop(:,:,2))
% end


% Make a figure of all the blue channels
% figure
% for i=1:num_files
%     img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
%     subplot(6,6,i)
%     colormap(gray)
%     imagesc(img_loop(:,:,3))
% end

% As we've established in the homework, the green channel is the most
% useful

% Intensity gradient seems to be an issue.  What happens if we simply try
% a simple threshold to pull out the brighter things?


% Make histograms of all the green channels
figure
for i=1:num_files
    img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
    img_loop = img_loop(:,:,2);
    subplot(6,6,i)
    colormap(gray)
    hist(img_loop(:), 200)
end

% Maybe top 20%??
% figure
% for i=1:num_files
%     img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
%     img_loop = img_loop(:,:,2);
%     thresh = .8*max(img_loop(:));
%     subplot(6,6,i)
%     colormap(gray)
%     imagesc(img_loop>thresh)
% end

%  Too high and we don't get much.  Too low and the intensity gradient is
%  an issue.

%  What if we try the trick we learned in class about subtracting a mean
%  filtered (or Gaussian filtered) image first?
% figure
for i=1:num_files
% fig_panel_count = 1;
% for i=[1:4, (num_files-3):num_files]
    img = double(imread(sprintf('%s/%s',path, files(i).name)));
%     img = img-imgaussfilt(img, 3);
    
    lower = 160;
    upper = 175;
    % img = 0.005*img_029(:,:,1)+1.55*img_029(:,:,2)+0.005*img_029(:,:,3);
    img = img(:,:,2);
    img(img<lower) = 0;
    img(img>upper) = 255;
    img(img>=lower & img<=upper) = (255/(upper-lower))*img(img>=lower & img<=upper) -255*lower/(upper-lower);
%     
%     figure
%     imagesc(img)
%     axis('image')
%     
    img(img<=250) = 0;
    img(img>250) = 1;
%     
%     figure
%     imagesc(img)
%     axis('image')
    
    CC = bwconncomp(img);

%     xmax = size(img,1);
%     ymax = size(img, 2);
%     seg_loc = 0*img;
% 
%     seg_label = 1;
%     [row,col] = find(img==1 & seg_loc==0, 1);
% 
%     while ~isempty(row)
% 
%         seg_label = seg_label+1;
%         old_reg_locx = [];
%         old_reg_locy = [];
%         reg_locx = row(1);
%         reg_locy = col(1);
% 
%         while length(reg_locx)~=length(old_reg_locx)  % suffices to only check x
%             old_reg_locx = reg_locx;
%             old_reg_locy = reg_locy;
%             % First create the "grown" coordinates
%             reg_locx = [old_reg_locx; old_reg_locx; old_reg_locx; old_reg_locx+1;...
%                 old_reg_locx-1];
%             reg_locy = [old_reg_locy; old_reg_locy-1; old_reg_locy+1; old_reg_locy;...
%                 old_reg_locy];
%             % remove values out of range
%             bad = reg_locx<1 | reg_locx>xmax | reg_locy<1 | reg_locy>ymax;
%             reg_locx = reg_locx(bad==0);
%             reg_locy = reg_locy(bad==0);
%             % remove duplicates 
%             coord_all = [reg_locx, reg_locy];
%             coord_all = unique(coord_all, 'rows');
%             % MATLAB can only extract values from a matrix according to multiply
%             % (x,y) locations if you first translate to the linear index
%             loc_linear = sub2ind(size(img), coord_all(:,1), coord_all(:,2));
%             seg_loc(loc_linear) = seg_label*img(loc_linear);
%             % Since we already converted img to 0s and 1s, the right hand side of
%             % the above will only create labels for the pixels that should be
%             % labeled
%             % Last, regenerate the x,y indices that made it to the end
%             [reg_locx, reg_locy] = find(seg_loc == seg_label);
%         end
%         [row,col] = find(img==1 & seg_loc==0, 1);
%     end
%     seg_label_list(i) = seg_label;
    seg_label_list(i) = CC.NumObjects;
    percentage_white(i) = length(find(img>0))/length(img(:));
    fig_list(i) = i;
end
figure
plot(fig_list,seg_label_list)
figure
plot(fig_list,percentage_white)

% Sometimes it is helpful to focus on a smaller subset.  I'll take first 4
% and last 4
% 
% figure
% fig_panel_count = 1;
% for i=[1:4, (num_files-3):num_files]
%     img_loop = double(imread(sprintf('%s/%s',path, files(i).name)));
%     img_loop = img_loop(:,:,2);
%     %img_loop_filt = img_loop-imgaussfilt(img_loop, 2);
%     img_loop_filt = img_loop-medfilt2(img_loop, [11,11]);
%     subplot(2,4,fig_panel_count) 
%     imagesc(img_loop_filt)
%     %hist(img_loop_filt(:), 200)
%     colormap(gray)
%     fig_panel_count = fig_panel_count+1;
% end
