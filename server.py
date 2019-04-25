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
class TServer (threading.Thread):
    def __init__ (self, socket, addr, recv_que, send_que):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = addr
        self.recv_que = recv_que
        self.send_que = send_que

    def run (self):
        strSend = ""
        recv_count = 0
        send_count = 0
        img_list = []
        while True:
            try:
                data = self.socket.recv(65536)
                if not data:
                    print('recieve data failed: {}'.format(recv_count))
                    break
                recv_count = recv_count + 1
                print('recieve data: {}'.format(recv_count))
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
                self.socket.close()
                print('socket closed')
                break
            '''
            data = self.socket.recv(65536)
            if not data:
                print('recieve data failed: {}'.format(recv_count))
                break
            recv_count = recv_count + 1
            print('recieve data success: {}'.format(recv_count))
            img = Image.open(BytesIO(base64.b64decode(data)))
            Image.save('C:\\Users\\P100\\NeoHand_server\\dataset'+str(recv_count)+'_recv.jpg')
            img = np.array(img).astype('int32')
            # append in que
            self.recv_que.append(img)

            # recv
            if recv_count % model_freq == 0:
                recv_count = 0
                model_result = self.func(self.recv_que)
                # send
                for i in range(model_freq):
                    self.socket.send(model_result[i+1].encode('ascii'))
                    self.recv_que.clear()
                    send_count = send_count + 1
                    print('send data: {}'.format(send_count))
            '''
            # filename = "D:/pictures/" + str(count) + ".jpg"
            # img.save(filename)

            # test print
            # print("output: picture " + str(count))
            # count += 1

def load_image_file():
    data_path = 'C:\\Users\\P100\\NeoHand_server\\dataset\\picture\\'
    buf = []
    img_name = data_path + 'webcam_1.jpg'
    buf.append(scipy.misc.imread(img_name))
    buf[0] = buf[0].astype('int32')
    return buf

def check_model_input(recv_que, send_que):
    #pool = mp.Pool(processes=10)
    run_model_process = []
    while True:
        if recv_que.qsize() > 0:
            inputbuf = recv_que.get()
            '''
            res = pool.apply_async(run_model, args=(inputbuf, send_que))
            outputbuf = res.get(timeout=1)
            send_que.put(outputbuf)
            '''
            p = mp.Process(target=run_model, args=(inputbuf, send_que))
            run_model_process.append(p)
            p.start()

def run_model(inputbuf, send_que):

    print('model start')
    ht = HandTrack('P100', model_freq)
    # setting variables
    ht.start()

    # open picture file, inputbuf type -> np.int32(H*W*3)
    #inputbuf = thread[0].load_image_file()

    # load image buffer to model
    ht.load_image_buf(inputbuf)

    # model running
    ht.hand_track_model()
    
    # outbuf type -> string list (7*63*num_images)
    outbuf = ht.write_result_buf()

    # write result
    #thread[0].write_result_file(outbuf)
    send_que.put(outbuf)
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
    #img = load_image_file()
    #recv_que.put(img)
    
    mp.Process(target=check_model_input, args=(recv_que, send_que)).start()
    while True:
        connect_socket, client_addr = server.accept()
        print('connected')
        TServer(connect_socket, client_addr, recv_que, send_que).start()
