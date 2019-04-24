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

model_freq = 1
class TServer (threading.Thread):
    def __init__ (self, socket, addr):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = addr

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
                print('recieve data success: {}'.format(recv_count))
                img = Image.open(BytesIO(base64.b64decode(data)))
                img = np.array(img).astype('int32')
                img_list.append(img)

                # recv
                if recv_count % model_freq == 0:
                    recv_count = 0
                    #model_result = self.func(recv_que)
                    recv_que.put(img_list)
                    img_list.clear()
                    model_process.start()
                model_process.join()
                # send
                for i in range(model_freq):
                    outdata = recv_que.get()
                    self.socket.send(outdata.encode('ascii'))
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

def handtrack(recv_que, send_que):
    # HandTrack('env name', image numbers)
    ht = HandTrack('P100', model_freq)

    # setting variables
    ht.start()

    # open picture file, inputbuf type -> np.int32(H*W*3)
    #inputbuf = thread[0].load_image_file()

    # load image buffer to model
    ht.load_image_buf(recv_que.get())

    # model running
    ht.hand_track_model()

    # outbuf type -> string list (7*63*num_images)
    outbuf = ht.write_result_buf()

    # write result
    #thread[0].write_result_file(outbuf)
    send_que.put(outbuf)

if __name__=='__main__':
    os.environ['GLOG_minloglevel'] = '2'
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    hostname = ''
    port = 555

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((hostname, port))
    server.listen(2)

    running = {}
    lock = threading.Lock()

    manager = mp.Manager()
    recv_que = mp.Queue()
    send_que = mp.Queue()
    model_process = mp.Process(target=handtrack, args=(recv_que, send_que))
    while True:
        connect_socket, client_addr = server.accept()
        print('connected')
        TServer(connect_socket, client_addr).start()