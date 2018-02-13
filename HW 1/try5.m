a=[-3:.1:3]
b=[-3:.1:3]
[A, B]=meshgrid(a,b);
whos A

figure
subplot(1,2,1)       
imagesc(A)
colorbar
title('A')

subplot(1,2,2)      
imagesc(B)
colorbar
title('B')

[testA, testB]=meshgrid(1:2, 1:3)

