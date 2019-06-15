# -*- coding: utf-8 -*-
from model.EvalRegNet_ImageSequence import HandTrack
import os
import socket
import threading as td
import base64
import re
import numpy as np
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import multiprocessing as mp
from queue import Queue
import scipy, scipy.misc
from PIL import ImageEnhance
import time

model_freq = 1
bb_offset = 25
width = 640
height = 480
recv_que_dic = {}
send_que_dic = {}
model_processes_dic = {}
client_number = 4

class TServer (td.Thread):
    def __init__ (self, socket, client_addr):
        td.Thread.__init__(self)
        self.socket = socket
        self.process_id = client_addr
        self.recv_que = recv_que_dic[self.process_id]
        self.send_que = send_que_dic[self.process_id]

    def run (self):
        send_count = 0
        recv_freq = 0
        recv_count_total = 0
        img_list = []
        while True:
            try:
                data = self.socket.recv(65536)
                if not data:
                    #print('** P: {} recieve data failed: {} **'.format(self.process_id, recv_count_total))
                    break
                recv_count_total = recv_count_total + 1
                recv_freq = recv_freq + 1
                #print('P: {} recieve freq: {}\trecieve count: {}'.format(self.process_id,recv_freq, recv_count_total))
                try:
                    img = Image.open(BytesIO(base64.b64decode(data)))
                except:
                    recv_freq = recv_freq - 1
                    recv_count_total = recv_count_total - 1
                    #print('** P: {} image format error... **'.format(self.process_id))
                    continue
                else:
                    #set contrast
                    #img = ImageEnhance.Contrast(img).enhance(15)
                    w, h = img.size
                    #print('P: {} recieve image size: {} * {}'.format(self.process_id,w, h))
                    if(width == w and height == h):
                        img = np.array(img).astype('int32')
                        img_list.append(img)
                    else:
                        recv_freq = recv_freq - 1
                        recv_count_total = recv_count_total - 1
                        #print('** P: {} image size error... **'.format(self.process_id))
                        continue

                    # recv
                    if recv_freq % model_freq == 0:
                        recv_freq = 0
                        inputbuf = img_list[:]
                        self.recv_que.put(inputbuf)
                        img_list.clear()
                    
                    while self.send_que.qsize() > 0:
                        outputbuf = self.send_que.get()
                        # send
                        for i in range(model_freq):
                            self.socket.send(outputbuf[i].encode('ascii'))
                            send_count = send_count + 1
                            #print('P: {} send data: {}'.format(self.process_id, send_count))
                            
                            # output buffer context check
                            #buf_split = buf.split(' ', len(buf))
                            #print('P: {} outputbuf size: {}'.format(self.process_id,len(buf_split)))
                            #print(buf)
            except:
                break
                
        self.socket.close()
        print('** P: {} socket closed **'.format(self.process_id))
        model_processes_dic[self.process_id].terminate()
        model_processes_dic[self.process_id].join()
        del model_processes_dic[self.process_id]
        del recv_que_dic[self.process_id]
        del send_que_dic[self.process_id]
        print('** P: {} Process closed process_number: {}**'.format(self.process_id, len(model_processes_dic)))

if __name__=='__main__':
    os.environ['GLOG_minloglevel'] = '2'
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    hostname = ''
    port = 555

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((hostname, port))
    server.listen(client_number)

    while True:
        print('server listen...')
        connect_socket, client_addr = server.accept()
        print('connected by {}'.format(client_addr))
        
        recv_que_dic = { client_addr : mp.Queue() }
        send_que_dic = { client_addr : mp.Queue() }
        model_processes_dic = { client_addr : HandTrack('', model_freq, bb_offset, recv_que_dic[client_addr], send_que_dic[client_addr], client_addr) }
        model_processes_dic[client_addr].start()
        # waiting process setup
        time.sleep(5)
        TServer(connect_socket, client_addr).start()