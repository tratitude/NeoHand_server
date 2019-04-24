from model.EvalRegNet_ImageSequence import HandTrack

import socket
import threading

import os
import base64
import re

from PIL import Image
from PIL import ImageFile
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True

hostname = ''
port = 555

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((hostname, port))
server.listen(5)

running = {}
lock = threading.Lock()

class TServer (threading.Thread):
    def __init__ (self, socket, addr):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = addr

    def run (self, func=None):
        strSend = ""
        count = 0
        while True:
            try:
                data = self.socket.recv(65536)
                if not data:
                    break

                img = Image.open(BytesIO(base64.b64decode(data)))

                if func==None:
                    strSend = ""
                else:
                    strSend = func(img)
                    self.socket.send(strSend)

            except:
                self.socket.close()
                break

            # filename = "D:/pictures/" + str(count) + ".jpg"
            # img.save(filename)

            # test print
            # print("output: picture " + str(count))
            # count += 1

while True:
    connect_socket, client_addr = server.accept()
    TServer(connect_socket, client_addr).start()
    
    
# server main function
thread = []
# HandTrack('env name', image numbers)
thread.append(HandTrack('fdmdkw', 1))

# setting variables
thread[0].start()

# open picture file, inputbuf type -> np.int32(H*W*3)
inputbuf = thread[0].load_image_file()

# load image buffer to model
thread[0].load_image_buf(inputbuf)

# model running
thread[0].hand_track_model()

# outbuf type -> string list (7*63*num_images)
outbuf = thread[0].write_result_buf()

# write result
thread[0].write_result(outbuf)
