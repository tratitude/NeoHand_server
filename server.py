from model.EvalRegNet_ImageSequence import HandTrack
import os
import socket
import threading
import base64
import re
import numpy as np
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import multiprocessing as mp
from queue import Queue
import scipy, scipy.misc

model_freq = 3
bb_offset = 75

class TServer (threading.Thread):
    def __init__ (self, socket, addr, recv_que, send_que):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = addr
        self.recv_que = recv_que
        self.send_que = send_que

    def run (self):
        send_count = 0
        recv_count = 0
        recv_count_total = 0
        img_list = []
        #threading.Thread(target=send, args=(self.send_que, self.socket)).start()
        while True:
            try:
                data = self.socket.recv(65536)
                if not data:
                    print('recieve data failed: {}'.format(recv_count))
                    break
                recv_count_total = recv_count_total + 1
                recv_count = recv_count + 1
                print('recieve freq: {}\trecieve count: {}'.format(recv_count, recv_count_total))
                img = Image.open(BytesIO(base64.b64decode(data)))
                img = np.array(img).astype('int32')
                img_list.append(img)

                # recv
                if recv_count % model_freq == 0:
                    recv_count = 0
                    #model_result = self.func(recv_que)
                    inputbuf = img_list[:]
                    self.recv_que.put(inputbuf)
                    img_list.clear()
                
                while self.send_que.qsize() > 0:
                    outputbuf = self.send_que.get()
                    # send
                    for i in range(model_freq):
                        self.socket.send(outputbuf[i].encode('ascii'))
                        send_count = send_count + 1
                        print('send data: {}'.format(send_count))
                
            except:
                # ignore exception
                '''
                self.socket.close()
                print('socket closed')
                break
                '''

def send(send_que, socket):
    send_count = 0
    while send_que.qsize() > 0:
        outputbuf = send_que.get()
        # send
        for i in range(model_freq):
            socket.send(outputbuf[i].encode('ascii'))
            send_count = send_count + 1
            print('send data: {}'.format(send_count))

def load_image_file():
    data_path = 'C:\\Users\\P100\\NeoHand_server\\dataset\\picture\\'
    buf = []
    img_name = data_path + 'webcam_1.jpg'
    buf.append(scipy.misc.imread(img_name))
    buf[0] = buf[0].astype('int32')
    return buf

def check_model_input(recv_que, send_que):
    # muti process by pool
    pool = mp.Pool(processes=4)
    result = []

    # muti process by shared memory
    #run_model_process = []

    while True:
        if recv_que.qsize() > 0:
            inputbuf = recv_que.get()
            result.append(pool.apply_async(run_model, args=(inputbuf,)))
        if len(result) > 0:
            for res in result:
                outputbuf = res.get()
                send_que.put(outputbuf)
            
            # muti process by shared memory
            '''
            p = mp.Process(target=run_model, args=(inputbuf, send_que))
            run_model_process.append(p)
            p.start()
            '''

# multi process by shared memory
def run_model(inputbuf, send_que):
# multi process by pool
#def run_model(inputbuf):

    # initialize object
    ht = HandTrack('fdmdkw', model_freq)
    
    # setting variables
    ht.start()
    print('model start')
    

    # open picture file, inputbuf type -> np.int32(H*W*3)
    #inputbuf = ht.load_image_file()

    # load image buffer to model
    ht.load_image_buf(inputbuf)

    # model running
    ht.hand_track_model()
    
    # outbuf type -> string list (7*63*num_images)
    outbuf = ht.write_result_buf()

    # write result
    #ht.write_result_file(outbuf)

    # use to mutlti processing by shared memory
    #send_que.put(outbuf)
    
    print('model finished')
    
    #return outbuf


if __name__=='__main__':
    os.environ['GLOG_minloglevel'] = '2'
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    hostname = ''
    port = 555

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((hostname, port))
    server.listen(2)

    recv_que = mp.Queue()
    send_que = mp.Queue()

    # test load image to shared memory
    #img = load_image_file()
    #recv_que.put(img)
    
    # multi process
    #mp.Process(target=check_model_input, args=(recv_que, send_que)).start()
    
    # initialize object
    ht = HandTrack('fdmdkw', model_freq, bb_offset, recv_que, send_que)
    ht.start()
    
    while True:
        print('server listen...')
        connect_socket, client_addr = server.accept()
        print('connected...')
        TServer(connect_socket, client_addr, recv_que, send_que).start()
        # error
        #threading.Thread(target=send, args=(send_que, connect_socket)).start()
