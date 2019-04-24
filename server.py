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

os.environ['GLOG_minloglevel'] = '2'
ImageFile.LOAD_TRUNCATED_IMAGES = True

hostname = ''
port = 555

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((hostname, port))
server.listen(5)

running = {}
lock = threading.Lock()

# run model frequency
model_freq = 3

class TServer (threading.Thread):
    def __init__ (self, socket, addr, func=None):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = addr
        self.func = func
        self.recv_que = []
        self.send_que = []

    def run (self):
        strSend = ""
        recv_count = 0
        send_count = 0
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
            except:
                self.socket.close()
                break

            # filename = "D:/pictures/" + str(count) + ".jpg"
            # img.save(filename)

            # test print
            # print("output: picture " + str(count))
            # count += 1

def handtrack(inputbuf):
    thread = []
    # HandTrack('env name', image numbers)
    thread.append(HandTrack('fdmdkw', model_freq))

    # setting variables
    thread[0].start()

    # open picture file, inputbuf type -> np.int32(H*W*3)
    #inputbuf = thread[0].load_image_file()

    # load image buffer to model
    thread[0].load_image_buf(inputbuf)

    # model running
    thread[0].hand_track_model()

    # outbuf type -> string list (7*63*num_images)
    outbuf = thread[0].write_result_buf()

    # write result
    #thread[0].write_result_file(outbuf)
    return outbuf

while True:
    connect_socket, client_addr = server.accept()
    TServer(connect_socket, client_addr, handtrack).start()