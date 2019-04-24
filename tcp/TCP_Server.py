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

    def run (self):
        
        count = 0
        while True:
            try:
                data = self.socket.recv(65536)
                if not data:
                    break

                img = Image.open(BytesIO(base64.b64decode(data)))


            except:
                self.socket.close()
                break

            filename = 'C:\\Users\\P100\\NeoHand_server\\dataset\\'+str(count)+'_recv.jpg'
            img.save(filename)

            # test print
            print("output: picture " + str(count))
            count += 1

while True:
    connect_socket, client_addr = server.accept()
    TServer(connect_socket, client_addr).start()
    