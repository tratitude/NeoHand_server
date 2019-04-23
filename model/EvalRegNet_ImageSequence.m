clear;
%% TODO: adapt these variables 
% add your matcaffe path here
addpath('C:\Users\Kellen\caffe\matlab');
% path to your images
data_path = 'C:\Users\Kellen\Pictures\dataset\webcam4\';
data_prefix = 'webcam';
data_suffix = '.jpg';
% file containing one line per image with: u_start, v_start, u_end, v_end of hand
% bounding box
BB_file = [data_path, 'boundbox.txt'];
% CNN stuff
GPU_id = 0;
net_base_path = '.';
net_architecture = 'RegNet_deploy.prototxt';
net_weights = 'RegNet_weights.caffemodel';
%% -----------------------------

crop_size = 128;
num_joints = 21;
o1_parent = [1, 1:4, 1, 6:8, 1, 10:12, 1, 14:16, 1, 18:20];
t_parent = [1, 1:4, 6:21];
i_parent = [1:5, 1, 6:8, 10:21];
m_parent = [1:9, 1, 10:12, 14:21];
r_parent = [1:13, 1, 14:16, 18:21];
l_parent = [1:17, 1, 18:20];

image_list = dir([data_path, data_prefix, '*', data_suffix]);
num_images = length(image_list);
all_pred3D = zeros(num_images, 3, num_joints);
all_pred2D = zeros(num_images, 3, num_joints);
BB_data = dlmread(BB_file);
if (size(BB_data, 1) ~= num_images)
    error('Bounding box file needs one line per image (u_start, v_start, u_end, v_end).');
end

%caffe.set_mode_gpu()
%caffe.set_device(GPU_id)
caffe.set_mode_cpu()
caffe.reset_all()

net = caffe.Net(fullfile(net_base_path, net_architecture), fullfile(net_base_path, net_weights), 'test');

for i=1:num_images
    %% read full image
    image_full = imread([data_path,image_list(i).name]);
    image_full_vis = double(image_full)/255;
    height = size(image_full, 1);
    width = size(image_full, 2);
    minBB_u = BB_data(i, 1);
    minBB_v = BB_data(i, 2);
    maxBB_u = BB_data(i, 3);
    maxBB_v = BB_data(i, 4);
    width_BB = maxBB_u - minBB_u + 1;
    height_BB = maxBB_v - minBB_v + 1;
    
    sidelength = max(width_BB, height_BB);
    tight_crop = single(zeros(sidelength, sidelength, 3));

    %% make BB square inside image
    if (width_BB > height_BB) % landscape
        minBB_v = minBB_v - floor((width_BB - height_BB)/2);
        maxBB_v = min(height, maxBB_v + ceil((width_BB - height_BB)/2));
        offset_h = max(0, -minBB_v + 1);
        minBB_v = max(1, minBB_v);
        height_BB = maxBB_v - minBB_v + 1;
        offset_w = 0;
    else % portrait
        minBB_u = minBB_u - floor((height_BB - width_BB)/2);
        maxBB_u = min(width, maxBB_u + ceil((height_BB - width_BB)/2));
        offset_w = max(0, -minBB_u + 1);
        minBB_u = max(1, minBB_u);
        width_BB = maxBB_u - minBB_u + 1;
        offset_h = 0;
    end

    %% fill crop
    endBB_u = offset_w+width_BB;
    endBB_v = offset_h+height_BB;
    tight_crop((offset_h+1):endBB_v,(offset_w+1):endBB_u,:) = image_full(minBB_v:maxBB_v, minBB_u:maxBB_u, :);
    % repeat last color at boundaries
    if (offset_w > 0)
        tight_crop(:,1:offset_w,:) = repmat(tight_crop(:,offset_w+1,:), [1, offset_w, 1]);
    end
    if (width_BB < sidelength)
        tight_crop(:,(endBB_u+1):sidelength,:) = repmat(tight_crop(:,endBB_u,:), [1, sidelength-endBB_u, 1]);
    end
    if (offset_h > 0)
        tight_crop(1:offset_h,:,:) = repmat(tight_crop(offset_h+1,:,:), [offset_h, 1, 1]);
    end
    if (height_BB < sidelength)
        tight_crop((endBB_v+1):sidelength,:,:) = repmat(tight_crop(endBB_v,:,:), [sidelength-endBB_v, 1, 1]);
    end

    %% resize and normalize
    tight_crop_sized = imresize(tight_crop, [crop_size, crop_size], 'bilinear');
    image_crop_vis = tight_crop_sized/255;

    % transform from [0,255] to [-1,1]
    tight_crop_sized = (tight_crop_sized/127.5) - 1;
    tight_crop_sized = permute(tight_crop_sized, [2 1 3]);

    % forward net
    pred = net.forward({tight_crop_sized});
    heatmaps = pred{1,1};
    pred_3D = pred{2,1};

    pred_3D = reshape(pred_3D, 3, []);
    all_pred3D(i, :, :) = pred_3D;

    % visualize skeleton in pred_3D
    figure(1); clf;
    %plot3([pred_3D(1,:); pred_3D(1,o1_parent)], [pred_3D(2,:); pred_3D(2,o1_parent)], [pred_3D(3,:); pred_3D(3,o1_parent)],'r','LineWidth',3);
    %t
    plot3([pred_3D(1,:); pred_3D(1,t_parent)], [pred_3D(2,:); pred_3D(2,t_parent)], [pred_3D(3,:); pred_3D(3,t_parent)],'r','LineWidth',3, 'Color', 'y');
    grid on; a5=gca; hold(a5, 'on');
    %i
    plot3([pred_3D(1,:); pred_3D(1,i_parent)], [pred_3D(2,:); pred_3D(2,i_parent)], [pred_3D(3,:); pred_3D(3,i_parent)],'r','LineWidth',3, 'Color', 'm');
    %m
    plot3([pred_3D(1,:); pred_3D(1,m_parent)], [pred_3D(2,:); pred_3D(2,m_parent)], [pred_3D(3,:); pred_3D(3,m_parent)],'r','LineWidth',3, 'Color', 'c');
    %r
    plot3([pred_3D(1,:); pred_3D(1,r_parent)], [pred_3D(2,:); pred_3D(2,r_parent)], [pred_3D(3,:); pred_3D(3,r_parent)],'r','LineWidth',3, 'Color', 'g');
    %l
    plot3([pred_3D(1,:); pred_3D(1,l_parent)], [pred_3D(2,:); pred_3D(2,l_parent)], [pred_3D(3,:); pred_3D(3,l_parent)],'r','LineWidth',3, 'Color', 'black');

    % visualize joint
    p1 = plot3(pred_3D(1,:), pred_3D(2,:), pred_3D(3,:), 'o');
    p1.MarkerSize = 10;
    p1.MarkerFaceColor = 'b';
    hold(a5, 'off');
    
    % capture frame 1
    frame = getframe(1);
    im = frame2im(frame);
    [A,map] = rgb2ind(im,256);
    filename = [data_path, 'result\',int2str(i),'_pred3D.gif'];
    imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
    
    resize_fact = sidelength / crop_size;
    % visualize 2D per joint
    for j=1:num_joints
        heat_j = heatmaps(:,:,j);
        heat_j = heat_j';
        heat_j_crop = imresize(heat_j, [crop_size, crop_size], 'bicubic');
        [conf, maxLoc] = max(heat_j_crop(:));
        [max_v, max_u] = ind2sub(size(heat_j_crop), maxLoc);

        % convert to u,v in orig image
        orig_BB_uv = bsxfun(@min, [width_BB; height_BB], bsxfun(@max, [1;1], round([max_u; max_v] * resize_fact - [offset_w; offset_h])));   
        orig_uv = [minBB_u; minBB_v] + orig_BB_uv;
        all_pred2D(i, 1:2, j) = orig_uv;
        all_pred2D(i, 3, j) = conf;

        image_full_vis = insertShape(image_full_vis, 'FilledCircle', [orig_uv',10], 'Color', 'red');
        %image_full_vis = insertShape(image_full_vis, 'Line', [[all_pred2D(i,1,o1_parent(j)),all_pred2D(i,2,o1_parent(j))], orig_uv'], 'Color', 'red');
        % t
        image_full_vis = insertShape(image_full_vis, 'Line', [[all_pred2D(i,1,t_parent(j)),all_pred2D(i,2,t_parent(j))], orig_uv'], 'Color', 'y');
        % i
        image_full_vis = insertShape(image_full_vis, 'Line', [[all_pred2D(i,1,i_parent(j)),all_pred2D(i,2,i_parent(j))], orig_uv'], 'Color', 'm');
        % m
        image_full_vis = insertShape(image_full_vis, 'Line', [[all_pred2D(i,1,m_parent(j)),all_pred2D(i,2,m_parent(j))], orig_uv'], 'Color', 'c');
        % r
        image_full_vis = insertShape(image_full_vis, 'Line', [[all_pred2D(i,1,r_parent(j)),all_pred2D(i,2,r_parent(j))], orig_uv'], 'Color', 'g');
        % l
        image_full_vis = insertShape(image_full_vis, 'Line', [[all_pred2D(i,1,l_parent(j)),all_pred2D(i,2,l_parent(j))], orig_uv'], 'Color', 'black');
        
        %{
        figure(2); imshow(image_full_vis);
        image_crop_vis = insertShape(image_crop_vis, 'FilledCircle', [max_u,max_v,3], 'Color', 'red');
        figure(3); imshowpair(image_crop_vis, heat_j_crop, 'montage');
        figure(4); imshowpair(image_crop_vis, heat_j_crop, 'blend');
        %}
    end
    
    %plot bounding box
    bb_pt = [[BB_data(i,1), BB_data(i,2)]; [BB_data(i,3), BB_data(i,2)]; [BB_data(i,3), BB_data(i,4)]; [BB_data(i,1), BB_data(i,4)]];
    bb_pt_line = [bb_pt(1,:),bb_pt(2,:); bb_pt(2,:),bb_pt(3,:); bb_pt(3,:),bb_pt(4,:); bb_pt(4,:),bb_pt(1,:)];
    image_full_vis = insertShape(image_full_vis, 'Line', bb_pt_line, 'Color', 'blue');
    result=[data_path, 'result\',int2str(i),'_withcam.jpg'];
    imwrite(image_full_vis,result);
    
    % visualize skeleton in all_pred2D
    figure(5); clf;
    %t
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,t_parent)],[2,21]), reshape([all_pred2D(i,2,:); all_pred2D(i,2,t_parent)], [2,21]), reshape([all_pred2D(i,3,:); all_pred2D(i,3,t_parent)],[2,21]),'r','LineWidth',3,'Color','y');
    a5=gca; hold(a5, 'on');
    %i
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,i_parent)],[2,21]), reshape([all_pred2D(i,2,:); all_pred2D(i,2,i_parent)], [2,21]), reshape([all_pred2D(i,3,:); all_pred2D(i,3,i_parent)],[2,21]),'r','LineWidth',3,'Color','m');
    %m
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,m_parent)],[2,21]), reshape([all_pred2D(i,2,:); all_pred2D(i,2,m_parent)], [2,21]), reshape([all_pred2D(i,3,:); all_pred2D(i,3,m_parent)],[2,21]),'r','LineWidth',3,'Color','c');
    %r
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,r_parent)],[2,21]), reshape([all_pred2D(i,2,:); all_pred2D(i,2,r_parent)], [2,21]), reshape([all_pred2D(i,3,:); all_pred2D(i,3,r_parent)],[2,21]),'r','LineWidth',3,'Color','g');
    %l
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,l_parent)],[2,21]), reshape([all_pred2D(i,2,:); all_pred2D(i,2,l_parent)], [2,21]), reshape([all_pred2D(i,3,:); all_pred2D(i,3,l_parent)],[2,21]),'r','LineWidth',3,'Color','black');
    % visualize joint
    p5 = plot3(reshape(all_pred2D(i,1,:),[1,21]), reshape(all_pred2D(i,2,:),[1,21]), reshape(all_pred2D(i,3,:),[1,21]), 'o');
    p5.MarkerSize = 10;
    p5.MarkerFaceColor = 'b';
    
    grid on;
    frame = getframe(5);
    im = frame2im(frame);
    [A,map] = rgb2ind(im,256);
    filename = [data_path, 'result\',int2str(i),'_pred2Din3D_.gif'];
    imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
    hold(a5, 'off');
    
    % write pred_3D
    outdir = [data_path, 'result\', int2str(i), '_pred3D.txt'];
    outfile = fopen(outdir, 'w');
    for out = pred_3D
        for outnum = 1:3
            fprintf(outfile, '%f ', out(outnum,:));
        end
    end
    fclose(outfile);
    
    % update bounding data
    if(i ~= num_images)
        bb_offset = 75;
        %minBB_u , minBB_v, maxBB_u, maxBB_v
        BB_data(i+1, 1) = max(1, min(all_pred2D(i,1,:)) - bb_offset);
        BB_data(i+1, 2) = max(1, min(all_pred2D(i,2,:)) - bb_offset);
        BB_data(i+1, 3) = min(width, max(all_pred2D(i,1,:)) + bb_offset);
        BB_data(i+1, 4) = min(height, max(all_pred2D(i,2,:)) + bb_offset);
    end
    fprintf("%d finished...\n", i);
end