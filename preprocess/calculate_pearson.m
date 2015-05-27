function calculate_pearson( data_file, affinity_file)
%
%Description :  calculate_affinity calculate the affinity (transition probability matrixes 
%Parameters:
%           data_file       - data file name
%           affinity_file   - affinity file name


 disp('Calculating affinity matrix...');
    data = load(data_file);
    affinity = load(affinity_file);
		P_tt = affinity.P_tt;
		P_ii = affinity.P_ii;
		P_ti = affinity.P_ti;
    image_fea = data.image_fea;
    text_fea = data.text_fea;
    co_image_fea = data.co_image_fea;
    co_text_fea = data.co_text_fea;

    text_num = size(text_fea,1);
    image_num = size(image_fea,1);
    co_image_num = size(co_image_fea,1);
    cii = zeros( co_image_num,image_num );
    co_text_num = size(co_text_fea,1);
    ctt = zeros( co_text_num,text_num );

     
    %  co-image to image
    disp('Calculating image to text transition matrix...');
    for idx1=1:co_image_num
        for idx2=1:image_num
            fea1 = co_image_fea( idx1, : );
            fea2 = image_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            cii(idx1,idx2) = sim;
        end
    end
    for idx1=1:co_text_num
        for idx2=1:text_num
            fea1 = co_text_fea( idx1, : );
            fea2 = text_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            ctt(idx1,idx2) = sim;
        end
    end
    save( affinity_file, 'P_tt', 'P_ii', 'P_ti', 'cii', 'ctt');
end

