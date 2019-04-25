import os
import sys
import math
import numpy as np
import numpy.matlib
import scipy, scipy.misc
import matplotlib.pyplot as plt
import json
import threading
import multiprocessing as mp
from queue import Queue

class HandTrack(mp.Process):
    def __init__(self, set_env, num_img, bb_offset, recv_que, send_que):
        mp.Process.__init__(self)
        # set_env is your env name in json
        self.set_env = set_env
        self.num_images = num_img
        self.set_parameters()
        self.recv_que = recv_que
        self.send_que = send_que
        # sliding boundbox offset
        self.bb_offset = bb_offset

    # setting from json
    def set_parameters(self):
        
        with open('C:\\Users\\Kellen\\NeoHand_server\\setting_py.json', 'r') as json_file:
            set_par = json.load(json_file)
            try:
                set_par[self.set_env]
            except ValueError:
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
                if self.append_caffe == 1:
                    sys.path.append(self.caffe_path)
        
                self.num_joints = 21
                # 建造零矩陣 *****矩陣可能長得跟matlab不一樣*****
                self.all_pred3D = np.zeros((self.num_images, 3, self.num_joints))
                self.all_pred2D = np.zeros((self.num_images, 3, self.num_joints))
                # image list
                self.image_full = np.zeros((self.num_images, 480, 640, 3), dtype=np.int32)
                # bounding box
                self.BB_data = np.zeros((4), dtype=np.int32)
        '''
        # caffe root path
        self.caffe_path = 'C:\\Users\\P100\\caffe\\python'
        # cpu mode or gpu mode
        self.mode = 0
        # path to your images
        self.data_path = 'C:\\Users\\P100\\NeoHand_server\\dataset\\'
        # 檔案位子
        self.net_base_path = 'C:\\Users\\P100\\NeoHand_server\\model\\'
        sys.path.append(self.caffe_path)
        self.num_joints = 21
        # 建造零矩陣 *****矩陣可能長得跟matlab不一樣*****
        self.all_pred3D = np.zeros((self.num_images, 3, self.num_joints))
        self.all_pred2D = np.zeros((self.num_images, 3, self.num_joints))
        # image list
        self.image_full = np.zeros((self.num_images, 480, 640, 3), dtype=np.int32)
        # bounding box
        self.BB_data = np.zeros((1, 4), dtype=np.int32)
        '''
    # output setting to json
    def out_parameters(self):
        set_par = {}
        # set your env name
        set_par[self.set_env] = []
        set_par[self.set_env].append({
            'append_caffe': 1,
            'caffe_path': 'C:\\Users\\P100\\caffe\\python',
            'mode': 0,
            'data_path': 'C:\\Users\\P100\\NeoHand_server\\dataset\\',
            'net_base_path': 'C:\\Users\\P100\\NeoHand_server\\model\\'
        })
        with open('setting_py.json', 'w') as outfile:
            json.dump(set_par, outfile)

    def hand_track_model(self):
        import caffe
        if self.mode == 1:
            # 模式設定為CPU
            caffe.set_mode_cpu()
        else:
            # 模式設定為GPU
            caffe.set_mode_gpu()
            caffe.set_device(0)
        # deploy檔案的路徑
        net_architecture = 'RegNet_deploy.prototxt'
        # 預訓練好的caffemodel的模型
        net_weights = 'RegNet_weights.caffemodel'
        # 參數
        crop_size = 128
        #o1_parent = [1, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]
        self.init_boundbox()

        # data number = BB data
        if self.BB_data.shape[0] != self.num_images:
            raise Exception("Bounding box file needs one line per image")

        net = caffe.Net((self.net_base_path+net_architecture), (self.net_base_path+net_weights), caffe.TEST)

        for i in range(self.num_images):
            #self.image_full = caffe.io.load_image(image)  # image_list->image   暫時沒測 mat無法顯示
            #self.image_full_vis = self.image_full
            #self.image_full = (self.image_full * 255).astype('int32')
            '''
            minBB_u = self.BB_data[i, 0]
            minBB_v = self.BB_data[i, 1]
            maxBB_u = self.BB_data[i, 2]
            maxBB_v = self.BB_data[i, 3]
            '''
            height = self.image_full[i].shape[0]
            width = self.image_full[i].shape[1]
            minBB_u, minBB_v, maxBB_u, maxBB_v = self.BB_data[i, :]
            width_BB = maxBB_u - minBB_u + 1
            height_BB = maxBB_v - minBB_v + 1

            sidelength = max(width_BB, height_BB)
            tight_crop = np.zeros((sidelength, sidelength, 3))

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
            tight_crop[offset_h:(endBB_v - 1), offset_w:(endBB_u - 1), :] = self.image_full[i, minBB_v - 1:maxBB_v - 1,
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
            # print(pred_3D[:, 0:3])
            self.all_pred3D[i - 1, :, :] = pred_3D

             # heatmap 數值誤差 會導致結果誤差
            heatmaps=np.reshape(heatmaps,(22,32,32))
            resize_fact = sidelength / crop_size
            for j in range(self.num_joints):
                heatj=heatmaps[j,:,:]
                heatj=np.transpose(heatj)
                heatj_crop=scipy.misc.imresize(heatj,(crop_size,crop_size),interp='bicubic',mode='F')
                conf=np.max(heatj_crop[:])
                maxLoc=np.argmax(heatj_crop)
                max_u=int(maxLoc/128)
                max_v=int(maxLoc%128)
                #orig_BB_uv = bsxfun(@min, [width_BB; height_BB], bsxfun(@max, [10;1], round([max_u; max_v] * resize_fact - [offset_w; offset_h])))
                BB_tmp=np.array([width_BB, height_BB]).astype(int)
                max_tmp=np.array([max_u,max_v])*resize_fact
                offset_tmp=np.array([offset_w,offset_h])
                BB_tmp2=((max_tmp-offset_tmp)+0.5).astype(int)
                orig_BB_uv = np.minimum(BB_tmp,np.maximum([1,1],BB_tmp2))
                #*************************************
                orig_uv=np.array([minBB_u, minBB_v])+ orig_BB_uv
                self.all_pred2D[i, 0:2, j] = orig_uv
                self.all_pred2D[i, 2, j] = conf

                self.update_boundbox(i)

    def run(self):
        import caffe
        if self.mode == 1:
            # 模式設定為CPU
            caffe.set_mode_cpu()
        else:
            # 模式設定為GPU
            caffe.set_mode_gpu()
            caffe.set_device(0)
        # deploy檔案的路徑
        net_architecture = 'RegNet_deploy.prototxt'
        # 預訓練好的caffemodel的模型
        net_weights = 'RegNet_weights.caffemodel'
        # 參數
        crop_size = 128
        #o1_parent = [1, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]
        
        self.BB_data[:] = [80, 1, 560, 480]

        # data number = BB data
        '''
        if self.BB_data.shape[0] != self.num_images:
            raise Exception("Bounding box file needs one line per image")
        '''

        net = caffe.Net((self.net_base_path+net_architecture), (self.net_base_path+net_weights), caffe.TEST)

        while True:
            if self.recv_que.qsize() > 0:
                inputbuf = self.recv_que.get()
                '''
                for buf in inputbuf:
                    self.image_full = np.append(self.image_full, buf, axis=0)
                '''
                self.image_full = np.append(self.image_full, inputbuf, axis=0)
                print('model start')
                for i in range(self.num_images):
                    #self.image_full = caffe.io.load_image(image)  # image_list->image   暫時沒測 mat無法顯示
                    #self.image_full_vis = self.image_full
                    #self.image_full = (self.image_full * 255).astype('int32')
                    '''
                    minBB_u = self.BB_data[i, 0]
                    minBB_v = self.BB_data[i, 1]
                    maxBB_u = self.BB_data[i, 2]
                    maxBB_v = self.BB_data[i, 3]
                    '''
                    height = self.image_full[i].shape[0]
                    width = self.image_full[i].shape[1]
                    minBB_u, minBB_v, maxBB_u, maxBB_v = self.BB_data[:]
                    width_BB = maxBB_u - minBB_u + 1
                    height_BB = maxBB_v - minBB_v + 1

                    sidelength = max(width_BB, height_BB)
                    tight_crop = np.zeros((sidelength, sidelength, 3))

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

                    tight_crop[offset_h:endBB_v , offset_w:endBB_u , :] = self.image_full[i, (minBB_v - 1):maxBB_v , (minBB_u - 1):maxBB_u , :]

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
                    # print(pred_3D[:, 0:3])
                    self.all_pred3D[i - 1, :, :] = pred_3D

                    # heatmap 數值誤差 會導致結果誤差
                    heatmaps=np.reshape(heatmaps,(22,32,32))
                    resize_fact = sidelength / crop_size
                    for j in range(self.num_joints):
                        heatj=heatmaps[j,:,:]
                        heatj=np.transpose(heatj)
                        heatj_crop=scipy.misc.imresize(heatj,(crop_size,crop_size),interp='bicubic',mode='F')
                        conf=np.max(heatj_crop[:])
                        maxLoc=np.argmax(heatj_crop)
                        max_u=int(maxLoc/128)
                        max_v=int(maxLoc%128)
                        #orig_BB_uv = bsxfun(@min, [width_BB; height_BB], bsxfun(@max, [10;1], round([max_u; max_v] * resize_fact - [offset_w; offset_h])))
                        BB_tmp=np.array([width_BB, height_BB]).astype(int)
                        max_tmp=np.array([max_u,max_v])*resize_fact
                        offset_tmp=np.array([offset_w,offset_h])
                        BB_tmp2=((max_tmp-offset_tmp)+0.5).astype(int)
                        orig_BB_uv = np.minimum(BB_tmp,np.maximum([1,1],BB_tmp2))
                        #*************************************
                        orig_uv=np.array([minBB_u, minBB_v])+ orig_BB_uv
                        self.all_pred2D[i, 0:2, j] = orig_uv
                        self.all_pred2D[i, 2, j] = conf
                        
                    self.update_boundbox(i)
            
                for i in range(self.num_images):
                    self.image_full = np.delete(self.image_full, 0, 0)
                outbuf = self.write_result_buf()
                self.send_que.put(outbuf)
                print('model push to send_que')


    # write pred2D to file
    def write_result2D(self):
        # path of pred_3D result
        buf_2D= [self.num_images]
        self.all_pred2D=np.swapaxes(self.all_pred2D,1,2)
        print(self.all_pred2D.shape)
        for i in range(self.num_images):
            strbuf = ''
            for k in range(self.num_joints):
                for j in range(3):
                    strbuf = strbuf + str(self.all_pred2D[i, k, j]) + ' '
            buf_2D.append(strbuf)
        self.pred3D_result = self.data_path + 'result_py\\'
        if not os.path.exists(self.pred3D_result):
            os.makedirs(self.pred3D_result)
        for i in range(self.num_images):
            fp = open(self.pred3D_result+str(i+1)+'_pred2D_py.txt', 'w')
            fp.write(buf_2D[i])
            fp.close()

    # write pred3D to file
    def write_result_file(self, buf):
        # path of pred_3D result
        self.pred3D_result = self.data_path + 'result_py\\'
        if not os.path.exists(self.pred3D_result):
            os.makedirs(self.pred3D_result)
        for i in range(self.num_images):
            fp = open(self.pred3D_result+str(i+1)+'_pred3D_py.txt', 'w')
            fp.write(buf[i])
            fp.close()

    # outbuf type -> string list (8*num_images)
    def write_result_buf(self):
        buf = []
        for i in range(self.num_images):
            strbuf = []
            for j in range(3):
                for k in range(self.num_joints):
                    # remove the last ' '
                    '''
                    if k+1 == self.num_joints:
                        strbuf = strbuf + '{:.3f}'.format(self.all_pred3D[i, j, k])
                    else:
                        strbuf = strbuf + '{:.3f} '.format(self.all_pred3D[i, j, k])
                    '''
                    strbuf.append('{:.3f}'.format(self.all_pred3D[i, j, k]))
            buf.append(' '.join(strbuf))
        return buf

    # buf is a np.int32(H*W*3)
    def load_image_buf(self, buf):
        for b in range(self.num_images):
            self.image_full[b, ...] = buf[b]

    def load_image_file(self):
        buf = []
        for i in range(self.num_images):
            img_name = self.data_path + 'webcam_'+str(1)+'.jpg'
            buf.append(scipy.misc.imread(img_name))
            buf[i] = buf[i].astype('int32')
        return buf

    def update_boundbox(self, i):
        height = self.image_full[i].shape[0]
        width = self.image_full[i].shape[1]
        
        tmp0 = np.max([1, np.min(self.all_pred2D[i, 0, :]) - self.bb_offset])
        self.BB_data[0] = tmp0.astype(np.int32)
        
        tmp1 = np.max([1, np.min(self.all_pred2D[i, 1, :]) - self.bb_offset])
        self.BB_data[1] = tmp1.astype(np.int32)
        
        tmp2 = np.min([width, np.max(self.all_pred2D[i, 0,:]) + self.bb_offset])
        self.BB_data[2] = tmp2.astype(np.int32)
        
        tmp3 = np.min([height, np.max(self.all_pred2D[i, 1,:]) + self.bb_offset])
        self.BB_data[3] = tmp3.astype(np.int32)
