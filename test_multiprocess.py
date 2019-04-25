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


class handtrack(mp.Process):
    def __init__(self, recv_que, send_que):
        super(handtrack, self).__init__()
        self.recv_que = recv_que
        self.send_que = send_que
    def run(self):
        run_model_processes = []
        while True: 
            if self.recv_que.qsize() > 0:
                inputbuf = self.recv_que.get()
                print('model start')
                p = mp.Process(target=self.run_model, args=(inputbuf))
                run_model_processes.append(p)
                p.start()
    def run_model(self, inputbuf):
        ht = HandTrack('P100', model_freq)

        # setting variables
        ht.start()

        # open picture file, inputbuf type -> np.int32(H*W*3)
        #inputbuf = thread[0].load_image_file()

        # load image buffer to model
        ht.load_image_buf(inputbuf)

        # model running
        ht.hand_track_model()
        print('model finished')
        # outbuf type -> string list (7*63*num_images)
        outbuf = ht.write_result_buf()

        # write result
        #thread[0].write_result_file(outbuf)
        self.send_que.put(outbuf)
    

if __name__=='__main__':
    os.environ['GLOG_minloglevel'] = '2'

    recv_que = Queue()
    send_que = Queue()
    handtrack(recv_que, send_que).start()
    