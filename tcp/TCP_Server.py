import socket
import threading
import os
import base64
import re

from PIL import Image
from PIL import ImageFile
from io import BytesIO

hostname = ''
port = 555

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((hostname, port))
server.listen(5)

running = {}
lock = threading.Lock()

ImageFile.LOAD_TRUNCATED_IMAGES = True

def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img

class TServer (threading.Thread):
    def __init__ (self, socket, addr):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = addr

    def run (self):

        
        count = 0
        while True:
            filename = "D:/pictures/" + str(count) + ".jpg"
            data = self.socket.recv(65536)
            if not data:
                break
            # with open(filename, "wb") as file:
            #     file.write(base64.decodebytes(data))
            
            img = Image.open(BytesIO(base64.b64decode(data)))
            img.save(filename)
            print("output: picture " + str(count))
            count += 1

        self.socket.close()

        # length = self.socket.recv(1024)
        # print(str(receive, encoding='gbk'))
        # self.socket.send(bytes('hi\n'))
        # self.socket.close()

    def recvall (self, msgLen):
        msg = ""
        bytesRead = 0

        while bytesRead < msgLen:
            chunk = self.socket.recv(msgLen - bytesRead)

            if chunk == "": break
            bytesRead += len(chunk)
            msg += chunk

            # if "\r\n" in msg: break
        
        return msg

while True:
    connect_socket, client_addr = server.accept()
    TServer(connect_socket, client_addr).start()
    