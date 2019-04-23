import os
import sys
import math
import numpy as np
import numpy.matlib
import scipy
import matplotlib.pyplot as plt
import json
import threading


class HandTrack(threading.Thread):
    def __init__(self, set_env):
        threading.Thread.__init__(self)
        # set_env is your env name in json
        self.set_env = set_env
        self.out_parameters()
        self.set_parameters()

    # setting from json
    def set_parameters(self):
        with open('setting_py.json', 'r') as json_file:
            set_par = json.load(json_file)
            try:
                set_par[self.set_env]
            except KeyError:
                self.out_parameters()
                self.set_parameters()
            else:
                for par in set_par[self.set_env]:
                    # need to append caffe or not
                    self.append_caffe = par['append_caffe']
                    # caffe root path
                    self.caffe_path = par['caffe_path']
                    # cpu mode or gpu mode
                    self.mode = par['mode']
                    # path to your images
                    self.data_path = par['data_path']
                    # 檔案位子
                    self.net_base_path = par['net_base_path']

    # output setting to json
    def out_parameters(self):
        set_par = {}
        # set your env name
        set_par[self.set_env] = []
        set_par[self.set_env].append({
            'append_caffe': 0,
            'caffe_path': '',
            'mode': 1,
            'data_path': 'C:\\Users\\Kellen\\Pictures\\dataset\\webcam5\\',
            'net_base_path': 'C:\\Users\\Kellen\\NeoHand_server\\model\\'
        })
        with open('setting_py.json', 'w') as outfile:
            json.dump(set_par, outfile)

    def hand_track_model(self):
        if self.append_caffe == 1:
            sys.path.append(self.caffe_path)
        import caffe

        if self.mode == 1:
            # 模式設定為CPU
            caffe.set_mode_cpu()
        else:
            # 模式設定為GPU
            caffe.set_mode_gpu()

        # path of pred_3D result
        self.pred3D_result = self.data_path + 'result_py\\'
        if not os.path.exists(self.pred3D_result):
            os.makedirs(self.pred3D_result)
        # deploy檔案的路徑
        net_architecture = 'RegNet_deploy.prototxt'
        # 預訓練好的caffemodel的模型
        net_weights = 'RegNet_weights.caffemodel'
        # Bounding box
        BB_file = self.data_path + 'boundbox.txt'
        # 測試圖片的路徑 *****檔名要改*****
        image = self.data_path + 'webcam_0.jpg'  # image_list->image
        # 參數
        crop_size = 128
        num_joints = 21
        o1_parent = [1, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]
        # 一次一張
        self.num_images = 1
        # 建造零矩陣 *****矩陣可能長得跟matlab不一樣*****
        self.all_pred3D = np.zeros(shape=(self.num_images, 3, num_joints))
        self.all_pred2D = np.zeros(shape=(self.num_images, 3, num_joints))
        # dlmread 可抓多行輸入
        BB_predata = open(BB_file, 'r')
        temp = BB_predata.readlines()
        for i in range(0, temp.__len__(), 1):
            BB_data = []
            for word in temp[i].split():
                word = word.strip(' ')
                BB_data.append(int(word))
        BB_data = np.atleast_2d(BB_data)

        # data number = BB data
        if np.array(BB_data).shape[0] != self.num_images:
            raise Exception("Bounding box file needs one line per image")

        net = caffe.Net((self.net_base_path+net_architecture), (self.net_base_path+net_weights), caffe.TEST)

        for i in range(self.num_images):
            image_full = caffe.io.load_image(image)  # image_list->image   暫時沒測 mat無法顯示
            image_full_vis = image_full
            image_full = (image_full * 255).astype('int32')
            height = image_full.shape[0]
            width = image_full.shape[1]
            minBB_u = BB_data[i - 1][0]
            minBB_v = BB_data[i - 1][1]
            maxBB_u = BB_data[i - 1][2]
            maxBB_v = BB_data[i - 1][3]
            width_BB = maxBB_u - minBB_u + 1
            height_BB = maxBB_v - minBB_v + 1

            sidelength = max(width_BB, height_BB)
            tight_crop = np.zeros(shape=(sidelength, sidelength, 3))

            if width_BB > height_BB:
                minBB_v = minBB_v - math.floor((width_BB - height_BB) / 2)
                maxBB_v = min(height, maxBB_v + math.ceil((width_BB - height_BB) / 2))
                offset_h = max(0, -minBB_v + 1)
                minBB_v = max(1, minBB_v)
                height_BB = maxBB_v - minBB_v + 1
                offset_w = 0
            else:
                minBB_u = minBB_u - math.floor((height_BB - width_BB) / 2)
                maxBB_u = min(width, maxBB_u + math.ceil((height_BB - width_BB) / 2))
                offset_w = max(0, -minBB_u + 1)
                minBB_u = max(1, minBB_u)
                width_BB = maxBB_u - minBB_u + 1
                offset_h = 0
            # fill crop
            endBB_u = offset_w + width_BB
            endBB_v = offset_h + height_BB

            #   matlab的index從1開始，python從0--->往左移一格
            tight_crop[offset_h:(endBB_v - 1), offset_w:(endBB_u - 1), :] = image_full[minBB_v - 1:maxBB_v - 1,
                                                                            minBB_u - 1:maxBB_u - 1, :]

            # repeat last color at boundaries
            if offset_w > 0:
                tight_crop[:, 0:offset_w, :] = np.matlib.tile(tight_crop[:, offset_w, :], [1, offset_w, 1])

            if (width_BB < sidelength):
                tight_crop[:, endBB_u:sidelength, :] = np.matlib.tile(tight_crop[:, endBB_u - 1, :],
                                                                    [1, sidelength - endBB_u, 1])

            if (offset_h > 0):
                tight_crop[0:offset_h, :, :] = np.matlib.tile(tight_crop[offset_h, :, :], [offset_h, 1, 1])

            if (height_BB < sidelength):
                tight_crop[endBB_v:sidelength, :, :] = np.matlib.tile(tight_crop[endBB_v - 1, :, :],
                                                                    [sidelength - endBB_v, 1, 1])

            ## resize and normalize
            tight_crop_sized = scipy.misc.imresize(tight_crop, (crop_size, crop_size), interp='bilinear',mode='RGB')
            image_crop_vis = tight_crop_sized / 255

            # transform from [0,255] to [-1,1]
            tight_crop_sized = (tight_crop_sized / 127.5) - 1
            tight_crop_sized = np.transpose(tight_crop_sized, (1, 0, 2))
            # forward net
            # *******************
            tight_crop_sized = tight_crop_sized.swapaxes(0, 2)
            # tight_crop_sized = tight_crop_sized.swapaxes(1, 2)
            tight_crop_sized = tight_crop_sized[np.newaxis, :]
            net.blobs['color_crop'].data[...] = tight_crop_sized
            pred = net.forward()
            # *******************
            heatmaps = pred['heatmap_final']
            pred_3D = pred['joints3D_final_vec']

            pred_3D = np.reshape(pred_3D, (3, -1))
            #print(pred_3D[:, 0:3])
            self.all_pred3D[i - 1, :, :] = pred_3D
    
    def model_result(self):
        for i in range(self.num_images):
            for out_1 in self.all_pred3D:
                fp = open(self.pred3D_result+str(i)+'_pred3D_py.txt', 'w')
                for out_2 in out_1:
                    for out_3 in out_2:
                        fp.write('{:0.6} '.format(out_3))
                fp.close()


thread = []
thread.append(HandTrack('fdmdkw'))
thread[0].start()
thread[0].hand_track_model()
thread[0].model_result()
