% KNN illustration

test = load('/Users/sparshagarwal/OneDrive/Spring 2018/CS 567/Medical-Imaging/HW 5/train.txt', '-ascii');
train = load('/Users/sparshagarwal/OneDrive/Spring 2018/CS 567/Medical-Imaging/HW 5/train.txt', '-ascii');

% train(1,1) 
% imagesc(reshape(train(1,2:end), 16,16)')

% for i=1:7291
    train_feat(:,:) = train(:,2:257);
    train_labels(:) = train(:,1);
    test_feat(:,:) = test(:,2:257);
    test_labels(:) = test(:,1);    
% end

%%

k = 256;

npairs = length(test_labels);
ntrain = size(train_feat, 1);

pred = zeros(npairs);

for i=1:npairs
    dist = sqrt(sum((ones(ntrain,1)*test_feat(i,:)-train_feat).^2, 2));
    [reord, ord] = sort(dist);
    knn=train_labels(ord(1:k));
    for j = 1:256
        p_g(j) = mean(knn == j);
    end
    sorted_prob = sort(p_g,'descend');
    pred(i) = sorted_prob(1,1);
end
hist(pred)

