function  HetTransfer2(data_file, similarity_file)
% Experiment: the main function used to compare all the online
% algorithms
%--------------------------------------------------------------------------
% Input:
%      dataset_name, name of the dataset, e.g. 'birds-food'
%
% Output:
%      a table containing the accuracies, the numbers of support vectors,
%      the running times of all the online learning algorithms on the
%      inputed datasets
%      a figure for the online average accuracies of all the online
%      learning algorithms
%      a figure for the online numbers of SVs of all the online learning
%      algorithms
%      a figure for the online running time of all the online learning
%      algorithms
%--------------------------------------------------------------------------

%load dataset
load(sprintf('%s', data_file));
%load(sprintf('%s','boats_toy'));
%load(sprintf('%s','flowers_tree'));
%load(sprintf('%s','vehicle_tree'));

%=========================================
load(sprintf('data/%s', similarity_file));
%load(sprintf('%s','boats_toy_sim'));
%load(sprintf('%s','flowers_tree_sim'));
%load(sprintf('%s','vehicle_tree_sim'));
P_it = P_ti';
%=========================================

num_new = 500;
options.Number_old = 0;

for i = 1 : 100
    ID_new(i,:) = randperm(num_new);
end

Y = image_gnd(1:num_new,:);
Y = full(Y);
X = image_fea_2(1:num_new,:);

[n,d] = size(X);
%[n,d] = size(image_fea);
% set parameters
options.C   = 5;
options.sigma = 4;
options.sigma2 = 8;
options.t_tick = round(size(ID_new,2)/10);

%====================================
options.K = 100;
%====================================

%%
m = size(ID_new,2);
options.beta=sqrt(m)/(sqrt(m)+sqrt(2*log(2)));
%ID_old = 1:n-m;



% scale
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);


P = sum(X.*X,2);
P = full(P);
disp('Pre-computing kernel matrix...');
Kernel = exp(-(repmat(P',n,1) + repmat(P,1,n)- 2*X*X')/(2*options.sigma2^2));
%K = X*X';


%% run experiments:
for i=1:size(ID_new,1),
    fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_new(i, :);
    
    %1. PA-I
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA2_K_M(Y,Kernel,options,ID);
    nSV_PA2(i) = length(classifier.SV);
    err_PA2(i) = err_count;
    time_PA2(i) = run_time;
    mistakes_list_PA2(i,:) = mistakes;
    
%     %2. HetTransfer-I-fixed
%     [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HetOTL12fixed(Y,Kernel,P_it,text_gnd,options,ID);
%     nSV_HT12fixed(i) = length(classifier.SV1)+length(classifier.SV2);
%     err_HT12fixed(i) = err_count;
%     time_HT12fixed(i) = run_time;
%     mistakes_list_HT12fixed(i,:) = mistakes;
%     
%     %3. HetTransfer-II-fixed
%     [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HetOTL22fixed(Y,Kernel,P_it,text_gnd,options,ID);
%     nSV_HT22fixed(i) = length(classifier.SV1)+length(classifier.SV2);
%     err_HT22fixed(i) = err_count;
%     time_HT22fixed(i) = run_time;
%     mistakes_list_HT22fixed(i,:) = mistakes;
% 
%     %4. HetTransfer-I
%     [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HetOTL12(Y,Kernel,P_it,text_gnd,options,ID);
%     nSV_HT12(i) = length(classifier.SV1)+length(classifier.SV2);
%     err_HT12(i) = err_count;
%     time_HT12(i) = run_time;
%     mistakes_list_HT12(i,:) = mistakes;
%     
%     %5. HetTransfer-II
%     [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HetOTL22(Y,Kernel,P_it,text_gnd,options,ID);
%     nSV_HT22(i) = length(classifier.SV1)+length(classifier.SV2);
%     err_HT22(i) = err_count;
%     time_HT22(i) = run_time;
%     mistakes_list_HT22(i,:) = mistakes;
%     

end


stat_file = sprintf('stat/%s-stat', data_file);
%save(stat_file, 'err_PA2', 'time_PA2', 'mistakes_list_PA2', 'err_HT12fixed', 'time_HT12fixed', 'mistakes_list_HT12fixed', 'err_HT22fixed', 'time_HT22fixed', 'mistakes_list_HT22fixed', 'err_HT12', 'time_HT12', 'mistakes_list_HT12', 'err_HT22', 'time_HT22', 'mistakes_list_HT22');
save('stat-birds-food-2', 'err_PA2', 'mistakes_list_PA2');


fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'number of mistakes, cpu running time\n');
fprintf(1,'PA             %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA2)/m*100,  std(err_PA2)/m*100, mean(time_PA2), std(time_PA2));
%fprintf(1,'OHT-I-fixed  %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_HT12fixed)/m*100,   std(err_HT12fixed)/m*100, mean(time_HT12fixed), std(time_HT12fixed));
%fprintf(1,'OHT-II-fixed  %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_HT22fixed)/m*100,   std(err_HT22fixed)/m*100, mean(time_HT22fixed), std(time_HT22fixed));
%fprintf(1,'OHT-I  %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_HT12)/m*100,   std(err_HT12)/m*100, mean(time_HT12), std(time_HT12));
%fprintf(1,'OHT-II  %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_HT22)/m*100,   std(err_HT22)/m*100, mean(time_HT22), std(time_HT22));
fprintf(1,'-------------------------------------------------------------------------------\n');


