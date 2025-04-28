input_folder = 'raw_mat_files';
output_folder = 'superSet';
file_list = dir(fullfile(input_folder, '*.mat'));
for i = 1:numel(file_list)
    file_name = file_list(i).name;
    file_path = fullfile(input_folder, file_name);
    
    mat_data = load(file_path);  
    cjdata = mat_data.cjdata;
    
    im1 = double(cjdata.image); 
    min1 = min(im1(:));
    max1 = max(im1(:));
    im = uint8(255/(max1-min1)*(im1-min1));
    label = cjdata.label;  
    
    label_folder = fullfile(output_folder, num2str(label));
    if ~exist(label_folder, 'dir')
        mkdir(label_folder); 
    end
    
    [~, file_name_base, ~] = fileparts(file_name);
    output_file_path = fullfile(label_folder, strcat(file_name_base, '.jpg'));
    
    imwrite(im, output_file_path);  
all_labels = zeros(1, numel(file_list));

for i = 1:numel(file_list)
    mat_data = load(fullfile(input_folder, file_list(i).name));
    if isfield(mat_data, 'cjdata') && isfield(mat_data.cjdata, 'label')
        all_labels(i) = mat_data.cjdata.label;
    end
end
end