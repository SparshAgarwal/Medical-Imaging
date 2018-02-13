img = imread('img_001.ppm'); 
red = img(:,:,1); % Red channel
whos red
img_new = zeros(length(red(:,1)), length(red));
whos img_new
for col = 2:length(red)-1
  for row = 2:length(red(:,1))-1
      img_new(row, col) = double((red(row-1,col-1)+red(row,col-1)+red(row+1,col-1)+red(row-1,col)+red(row,col)+red(row+1,col)+red(row-1,col+1)+red(row,col+1)+red(row+1,col+1))/9);
  end
end

red(1:5,1:5)
img_new(1:5,1:5)

figure
subplot(1,2,1)       
imshow(red)
title('Before')

subplot(1,2,2)      
imshow(img_new)    
title('After')