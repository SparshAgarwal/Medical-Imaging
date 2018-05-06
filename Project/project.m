% TODO:
% Process Image
% Normalize in all
% Normalization does not work because if sd is zero, matrix cell becomes NaN aND CALCULATING INVERSE BECOMES DIFFICULT
% Check LDA 
% Check KNN
% Check logistic
% check for 2,3,6 fold of cross validation (same fold in all three classifiers??)
    % Check for 2:9 Ks in KNN
        % run best cases for 10 times and get average (we dont have to do this in code???)
% check all with 1,2 labels (best fold number and k)
    % run best cases for 10 times and get average (we dont have to do this in code???)

%%
%Setup: Image loading and fold creation
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

%NotDistributed data
notdist_files_path = '/Users/sparshagarwal/OneDrive/Spring 2018/CS 567/Medical-Imaging/Project/distributed';
notdist_files = dir(sprintf('%s/img*.ppm',notdist_files_path));
notdist_num_files = length(notdist_files);
% for i=1:notdist_num_files
%     notdist_img(i,:,:,:) = double(imread(sprintf('%s/%s',notdist_files_path, notdist_files(i).name)));
% end
notdist_labels = [zeros(1,18),ones(1,18)];


%%
%Processing : Image processing to extract 2 features

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

% process notdistributed data
notdist_feat1 = zeros(1,notdist_num_files);
notdist_feat2 = zeros(1,notdist_num_files);
for j=1:notdist_num_files
    notdist_img = double(imread(sprintf('%s/%s',notdist_files_path, dist_files(j).name)));
    one  = notdist_img(:,:,1);
    two = notdist_img(:,:,2);
    three = notdist_img(:,:,3);
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
%     one  = notdist_img(j,:,:,1);
%     two = notdist_img(j,:,:,2);
%     three = notdist_img(j,:,:,3);
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
%     % Count number of distinct spots and store this number in feature vector as a feature
    CC = bwconncomp(img);
    notdist_feat1(j) = CC.NumObjects; % Feature 1: Number of segments
    notdist_feat2(j) = length(find(img>0))/length(img(:));  % Feature 2: Percentage of image white after processing
end

%print 4 feature vetors
% figure
% scatter([1:dist_num_files],dist_feat1);
% figure
% scatter([1:dist_num_files],dist_feat2);
% figure
% scatter([1:notdist_num_files],notdist_feat1);
% figure
% scatter([1:notdist_num_files],notdist_feat2);
figure
plot(dist_feat1(dist_labels==0), dist_feat2(dist_labels==0), 'r.', 'markersize', 10)
hold on
plot(dist_feat1(dist_labels==1), dist_feat2(dist_labels==1), 'g.', 'markersize', 10)
hold off
figure
plot(notdist_feat1(notdist_labels==0), notdist_feat2(notdist_labels==0), 'r.', 'markersize', 10)
hold on
plot(notdist_feat1(notdist_labels==1), notdist_feat2(notdist_labels==1), 'g.', 'markersize', 10)
hold off


%%
% Section 1: Only distributed data

% CV stuff
% Creating Folds for CV
feat = [dist_feat1; dist_feat2]';
labels = dist_labels;

% numFolds fold CV
folds1= repmat([1:numFolds], 1, 18/numFolds);
folds1 = folds1(randperm(18)); %why do we find two separate folds and then combine them?
folds2 = repmat([1:numFolds], 1, 18/numFolds);
folds2 = folds2(randperm(18));
folds = [folds1, folds2];

%%
% Section 1, Classifier 1 : numFolds fold CV classification
   
% feat = [dist_feat1; dist_feat2]';
% labels = dist_labels;
% 
% % numFolds fold CV
% folds1= repmat([1:numFolds], 1, 18/numFolds);
% folds1 = folds1(randperm(18)); %why do we find two separate folds and then combine them?
% folds2 = repmat([1:numFolds], 1, 18/numFolds);
% folds2 = folds2(randperm(18));
% folds = [folds1, folds2];

% %knn with k=5
% k=5;

% varying k
for k = 3:2:9
    pred = zeros(1, 36);
    for fold=1:numFolds
        test = feat(folds==fold, :);
        train = feat(folds~=fold, :);
        labels_train = labels(folds ~= fold);
        ntest = size(test, 1);
        ntrain = size(train, 1);
        nfeat = size(train, 2);
        pred_test = zeros(1, ntest);
        % with correct scaling
        for n=1:nfeat
            mn_train = mean(train(n,:));
            sd_train = std(train(n,:));
            train(n,:) = (train(n,:)-mn_train)/sd_train;
            test(n,:) = (test(n,:)-mn_train)/sd_train;
        end
        for i=1:ntest
            dist_from_train = sqrt(sum((ones(ntrain,1)*test(i,:)-train).^2, 2));
            [reord, ord] = sort(dist_from_train);
            knn = labels_train(ord(1:k));
            p_g1 = mean(knn == 0);
            p_g2 = mean(knn == 1);
            if (p_g2<p_g1)
                pred_test(i)=0;
            elseif (p_g1<p_g2)
                pred_test(i)=1;
            else
                pred_test(i)=randperm(2,1)-1;
            end
        end
        pred(folds == fold) = pred_test;
    end

    match = labels == pred;
    dist_acc1_knn = mean(match(labels == 0));
    dist_acc2_knn = mean(match(labels == 1));
    fprintf('Distributed Class 0 (Healthy retina) accuracy using KNN(k = %d)= %f%%.\n',k,100*dist_acc1_knn);
    fprintf('Distributed Class 1 (Unhealthy retina) accuracy using KNN(k = %d)= %f%%.\n',k,100*dist_acc2_knn);
end


%%
% Section 1, Classifier 2 : numFolds fold CV classification using LDA

% feat = [dist_feat1; dist_feat2]';
% labels = dist_labels;
% 
% % numFolds fold CV
% folds1= repmat([1:numFolds], 1, 18/numFolds);
% folds1 = folds1(randperm(18));
% folds2 = repmat([1:numFolds], 1, 18/numFolds);
% folds2 = folds2(randperm(18));
% folds = [folds1, folds2];

pred = zeros(1, 36);

% blank discriminant function that I can reuse in loop, So input order is Sigma, mu, pi_hat, x!
d = inline('x''*inv(Sigma)*mu-.5*mu''*inv(Sigma)*mu+log(pi_hat)');

for fold=1:numFolds
    test = feat(folds == fold,:);  
    train = feat(folds ~= fold,:);
    labels_train = labels(folds ~= fold);
    ntest = size(test, 1);
    ntrain = size(train, 1);
    nfeat = size(train, 2);
    pred_test = zeros(1, ntest);
    % with correct scaling
    %Normalization does not work because if sd is zero, matrix cell becomes NaN aND CALCULATING INVERSE BECOMES DIFFICULT
%     for n=1:nfeat
%         mn_train = mean(train(n,:));
%         sd_train = std(train(n,:));
%         train(n,:) = (train(n,:)-mn_train)/sd_train;
%         test(n,:) = (test(n,:)-mn_train)/sd_train;
%     end 
    % Estimate parameters for discriminant for each class
    train_class1 = train(labels_train == 0,:);
    train_class2 = train(labels_train == 1,:);
    mu1 = mean(train_class1)';
    mu2 = mean(train_class2)';
    N = size(train, 1);
    k = 2;
    P = size(train, 2);
    % To compute Sigma, I first loop through to create within-class Sigma
    ntrain1  = size(train_class1, 1);
    train1Sigma = zeros(P,P);
    for train1ind = 1:ntrain1
        xi = train_class1(train1ind,:)';
        train1Sigma = train1Sigma + (xi - mu1)*(xi - mu1)';
    end
    ntrain2  = size(train_class2, 1);
    train2Sigma = zeros(P,P);
    for train2ind = 1:ntrain2
        xi = train_class2(train2ind,:)';
        train2Sigma = train2Sigma + (xi - mu2)*(xi - mu2)';
    end 
    Sigma = 1/(N-k)*(train1Sigma + train2Sigma);
    pi1_hat = ntrain1/(ntrain1 + ntrain2);
    pi2_hat = ntrain2/(ntrain1 + ntrain2);
    for i=1:ntest
        x_test_loop = test(i,:)';
        d1 = d(Sigma, mu1, pi1_hat, x_test_loop);
        d2 = d(Sigma, mu2, pi2_hat, x_test_loop);
        if (d2<d1)
            pred_test(i)=0;
        elseif (d1<d2)
            pred_test(i)=1;
        else
            pred_test(i)=randperm(2,1)-1; 
        end   
    end
    pred(folds == fold) = pred_test;
end

match = labels == pred;
dist_acc1_lda = mean(match(labels == 0));
dist_acc2_lda = mean(match(labels == 1));
fprintf('Distributed Class 0 (Healthy retina) accuracy using LDA= %f%%.\n',100*dist_acc1_lda);
fprintf('Distributed Class 1 (Unhealthy retina) accuracy using LDA= %f%%.\n',100*dist_acc2_lda);


%%
% Section 1, Classifier 3 : numFolds fold CV classification using Logistic regression

% feat = [dist_feat1; dist_feat2]';
% labels = dist_labels;
% 
% % numFolds fold CV
% folds1= repmat([1:numFolds], 1, 18/numFolds);
% folds1 = folds1(randperm(18));
% folds2 = repmat([1:numFolds], 1, 18/numFolds);
% folds2 = folds2(randperm(18));
% folds = [folds1, folds2];

pred = zeros(1, 36);

for fold=1:numFolds
    test = feat(folds == fold,:);  
    train = feat(folds ~= fold,:);
    labels_train = labels(folds ~= fold);
    ntest = size(test, 1);
    ntrain = size(train, 1);
    nfeat = size(train, 2);
    pred_test = zeros(1, ntest);
    % generally, data should be normalized.  You will do this int he homework
    % with correct scaling
    for n=1:nfeat
        mn_train = mean(train(n,:));
        sd_train = std(train(n,:));
        train(n,:) = (train(n,:)-mn_train)/sd_train;
        test(n,:) = (test(n,:)-mn_train)/sd_train;
    end
    % Train the classifier (logistic model fit)
    beta = glmfit(train, labels_train', 'binomial', 'link', 'logit');

    % Need to use the inverse logit to get the probabilities for test
    xb = [ones(size(test,1), 1), test]*beta;
    prob_test = exp(xb)./(1+exp(xb));
    pred_test = 1*prob_test>.5;

    pred(folds == fold) = pred_test;
end

match = labels == pred;
dist_acc1_logistic = mean(match(labels == 0));
dist_acc2_logistic = mean(match(labels == 1));
fprintf('Distributed Class 0 (Healthy retina) accuracy using Logistic regression= %f%%.\n',100*dist_acc1_logistic);
fprintf('Distributed Class 1 (Unhealthy retina) accuracy using Logistic regression= %f%%.\n',100*dist_acc2_logistic);


%%
% Section 2 : Distributed data as training and unknown data as test
% Section 2, Classifier 1 : Classification using KNN

train_feat = [dist_feat1; dist_feat2]';
train_labels = dist_labels;
test_feat = [notdist_feat1; notdist_feat2]';
test_labels = notdist_labels;

% %knn with k=5
% k=5;

% varying k
for k = 3:2:9
    pred = zeros(1, 36);
    ntest = size(test_feat, 1);
    ntrain = size(train_feat, 1);
    pred_test = zeros(1, ntest);
    nfeat = size(train_feat, 2);
    % with correct scaling
    for n=1:nfeat
        mn_train = mean(train_feat(n,:));
        sd_train = std(train_feat(n,:));
        train_feat(n,:) = (train_feat(n,:)-mn_train)/sd_train;
        test_feat(n,:) = (test_feat(n,:)-mn_train)/sd_train;
    end
    for i=1:ntest
        dist_from_train = sqrt(sum((ones(ntrain,1)*test_feat(i,:)-train_feat).^2, 2));
        [reord, ord] = sort(dist_from_train);
        knn = train_labels(ord(1:k));
        p_g1 = mean(knn == 0);
        p_g2 = mean(knn == 1);
        if (p_g2<p_g1)
            pred_test(i)=0;
        elseif (p_g1<p_g2)
            pred_test(i)=1;
        else
            pred_test(i)=randperm(2,1)-1;
        end
    end
    pred = pred_test';
    match = test_labels == pred;
    notdist_acc1_knn = mean(match(test_labels == 0));
    notdist_acc2_knn = mean(match(test_labels == 1));
    fprintf('NotDistributed Class 0 (Healthy retina) accuracy using KNN(k = %d)= %f%%.\n',k,100*notdist_acc1_knn);
    fprintf('NotDistributed Class 1 (Unhealthy retina) accuracy using KNN(k = %d)= %f%%.\n',k,100*notdist_acc2_knn);
end


%%
% Section 2, Classifier 2 : Classification using LDA

train_feat = [dist_feat1; dist_feat2]';
train_labels = dist_labels;
test_feat = [notdist_feat1; notdist_feat2]';
test_labels = notdist_labels;

pred = zeros(1, 36);

% blank discriminant function that I can reuse in loop, So input order is Sigma, mu, pi_hat, x!
d = inline('x''*inv(Sigma)*mu-.5*mu''*inv(Sigma)*mu+log(pi_hat)');

ntest = size(test_feat, 1);
ntrain = size(train_feat, 1);
nfeat = size(train_feat, 2);
pred_test = zeros(1, ntest);
% with correct scaling
%Normalization does not work because if sd is zero, matrix cell becomes NaN aND CALCULATING INVERSE BECOMES DIFFICULT
% for n=1:nfeat
%     mn_train = mean(train_feat(n,:));
%     sd_train = std(train_feat(n,:));
%     train_feat(n,:) = (train_feat(n,:)-mn_train)/sd_train;
%     test_feat(n,:) = (test_feat(n,:)-mn_train)/sd_train;
% end 
% Estimate parameters for discriminant for each class
train_class1 = train_feat(train_labels == 0,:);
train_class2 = train_feat(train_labels == 1, :);
mu1 = mean(train_class1)';
mu2 = mean(train_class2)';
N = size(train_feat, 1);
k = 2;
P = size(train_feat, 2);
% To compute Sigma, I first loop through to create within-class Sigma
ntrain1  = size(train_class1, 1);
train1Sigma = zeros(P,P);
for train1ind = 1:ntrain1
    xi = train_class1(train1ind,:)';
    train1Sigma = train1Sigma + (xi - mu1)*(xi - mu1)';
end
ntrain2  = size(train_class2, 1);
train2Sigma = zeros(P,P);
for train2ind = 1:ntrain2
    xi = train_class2(train2ind,:)';
    train2Sigma = train2Sigma + (xi - mu2)*(xi - mu2)';
end 
Sigma = 1/(N-k)*(train1Sigma + train2Sigma);
pi1_hat = ntrain1/(ntrain1 + ntrain2);
pi2_hat = ntrain2/(ntrain1 + ntrain2);
for i=1:ntest
    x_test_loop = test_feat(i,:)';
    d1 = d(Sigma, mu1, pi1_hat, x_test_loop);
    d2 = d(Sigma, mu2, pi2_hat, x_test_loop);
    if (d2<d1)
        pred_test(i)=0;
    elseif (d1<d2)
        pred_test(i)=1;
    else
        pred_test(i)=randperm(2,1)-1; 
    end   
end
pred = pred_test';

notdist_acc1_lda = mean(pred(test_labels ==0)==0);
notdist_acc2_lda = mean(pred(test_labels ==1)==1);
fprintf('NotDistributed Class 0 (Healthy retina) accuracy using LDA= %f%%.\n',100*notdist_acc1_lda);
fprintf('NotDistributed Class 1 (Unhealthy retina) accuracy using LDA= %f%%.\n',100*notdist_acc2_lda);


%%
% Section 2, Classifier 3 : Classification using Logistic regression

train_feat = [dist_feat1; dist_feat2]';
train_labels = dist_labels;
test_feat = [notdist_feat1; notdist_feat2]';
test_labels = notdist_labels;

% with correct scaling
pred = zeros(1, 36);

% generally, data should be normalized.  You will do this int he homework
ntest = size(test_feat, 1);
ntrain = size(train_feat, 1);
pred_test = zeros(1, ntest);
nfeat = size(train_feat, 2);
% with correct scaling
for n=1:nfeat
    mn_train = mean(train_feat(n,:));
    sd_train = std(train_feat(n,:));
    train_feat(n,:) = (train_feat(n,:)-mn_train)/sd_train;
    test_feat(n,:) = (test_feat(n,:)-mn_train)/sd_train;
end

% Train the classifier (logistic model fit)
beta = glmfit(train_feat, train_labels', 'binomial', 'link', 'logit');

% Need to use the inverse logit to get the probabilities for test
xb = [ones(size(test_feat,1), 1), test_feat]*beta;
prob_test = exp(xb)./(1+exp(xb));
pred_test = 1*prob_test>.5;

pred = pred_test';

match = test_labels == pred;
notdist_acc1_logistic = mean(match(test_labels == 0));
notdist_acc2_logistic = mean(match(test_labels == 1));
fprintf('NotDistributed Class 0 (Healthy retina) accuracy using Logistic regression= %f%%.\n',100*notdist_acc1_logistic);
fprintf('NotDistributed Class 1 (Unhealthy retina) accuracy using Logistic regression= %f%%.\n',100*notdist_acc2_logistic);


%%
% Results : 8 Outputs of accuracies, 2 classes in 2 classifiers in 2 sections

% dist_acc1_knn
% dist_acc2_knn
% dist_acc1_lda
% dist_acc2_lda
% dist_acc1_logistic
% dist_acc2_logistic
% notdist_acc1_knn
% notdist_acc2_knn
% notdist_acc1_lda 
% notdist_acc2_lda
% notdist_acc1_logistic
% notdist_acc2_logistic

