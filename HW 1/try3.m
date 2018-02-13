img1 = imread('img_001.ppm');
whos img1

figure
imshow(img1)
title('imshow')

figure
image(img1)
title('image')

figure
imagesc(img1)
title('imagesc')


grey_img1 = rgb2gray(img1);
axis('square') 

figure
subplot(1,3,1)       
imshow(grey_img1)
colorbar
title('imshow')

subplot(1,3,2)      
image(grey_img1)
colorbar
title('image')

subplot(1,3,3)      
imagesc(grey_img1)
colorbar
title('imagesc')