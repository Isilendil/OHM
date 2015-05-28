function feature2(data_file)



similarity_file = sprintf('%s-similarity', data_file);

load(sprintf('data/%s', data_file));
load(sprintf('data/%s', similarity_file));

image_fea_2 = zeros(600, size(text_fea, 2));
K = 100;
cii = cii';

for t = 1 : 600
	
	sim = zeros(1, K);
	
    similarity = cii(t,:);
  feature = zeros(K, size(text_fea, 2));

  for iter = 1 : K
	  max_value = 0;
	  max_index = 1;
	  for jter = 1 : length(similarity)
			if (similarity(jter) > max_value)
				max_value = similarity(jter);
				max_index = jter;
			end
		end
		sim(iter) = similarity(max_index);
		feature(iter,:) = co_text_fea(max_index,:);
		similarity(max_index) = 0;
	end

	sim = sim / sum(sim);
	image_fea_2(t,:) = sim * feature;

end

save(sprintf('data/%s-2', data_file), 'image_fea', 'image_gnd', 'text_fea', 'text_gnd', 'co_image_fea', 'co_text_fea', 'image_fea_2');


end

