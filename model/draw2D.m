clear;
data_path = 'C:\Users\P100\NeoHand_server\dataset\picture\result_py\';
data_prefix = 'pred3D_py';
data_suffix = '.txt';
num_joints = 21;

o1_parent = [1, 1:4, 1, 6:8, 1, 10:12, 1, 14:16, 1, 18:20];
t_parent = [1, 1:4, 6:21];
i_parent = [1:5, 1, 6:8, 10:21];
m_parent = [1:9, 1, 10:12, 14:21];
r_parent = [1:13, 1, 14:16, 18:21];
l_parent = [1:17, 1, 18:20];

image_list = dir([data_path,'*', data_prefix, '*', data_suffix]);
num_images = length(image_list);

all_pred2D = zeros(num_images, 3, num_joints);
all_pred2D = zeros(num_images, 3, num_joints);

for i=1:num_images
    dataID = fopen([data_path,image_list(i).name], 'r');
    num = fscanf(dataID, '%f', [3,21]);
    all_pred2D(i,:,:) = num;
    fclose(dataID);

    for x=1:21
      for c=1:3
            if(c==1)
              all_pred2D[i,c,x]=all_pred2D[i,c,x]*320-320;
            else
              all_pred2D[i,c,x]=all_pred2D[i,c,x]*240-240;
            end
    end
    
    % visualize skeleton in all_pred2D
    figure(5); clf;
    %t
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,t_parent)],[2,21]),
          reshape([all_pred2D(i,2,:); all_pred2D(i,2,t_parent)], [2,21]),
          reshape([all_pred2D(i,3,:); all_pred2D(i,3,t_parent)],[2,21]),'r','LineWidth',3,'Color','y');
    a5=gca; hold(a5, 'on');
    %i
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,i_parent)],[2,21]),
          reshape([all_pred2D(i,2,:); all_pred2D(i,2,i_parent)], [2,21]),
          reshape([all_pred2D(i,3,:); all_pred2D(i,3,i_parent)],[2,21]),'r','LineWidth',3,'Color','m');
    %m
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,m_parent)],[2,21]),
          reshape([all_pred2D(i,2,:); all_pred2D(i,2,m_parent)], [2,21]),
          reshape([all_pred2D(i,3,:); all_pred2D(i,3,m_parent)],[2,21]),'r','LineWidth',3,'Color','c');
    %r
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,r_parent)],[2,21]), 
          reshape([all_pred2D(i,2,:); all_pred2D(i,2,r_parent)], [2,21]),
          reshape([all_pred2D(i,3,:); all_pred2D(i,3,r_parent)],[2,21]),'r','LineWidth',3,'Color','g');
    %l
    plot3(reshape([all_pred2D(i,1,:); all_pred2D(i,1,l_parent)],[2,21]),
          reshape([all_pred2D(i,2,:); all_pred2D(i,2,l_parent)], [2,21]),
          reshape([all_pred2D(i,3,:); all_pred2D(i,3,l_parent)],[2,21]),'r','LineWidth',3,'Color','black');
    % visualize joint
    p5 = plot3(reshape(all_pred2D(i,1,:),[1,21]),
               reshape(all_pred2D(i,2,:),[1,21]),
               reshape(all_pred2D(i,3,:),[1,21]), 'o');
    p5.MarkerSize = 10;
    p5.MarkerFaceColor = 'b';
end