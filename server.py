from EvalRegNet_ImageSequence import HandTrack

import socket
import threading

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
        print(self.address)
        receive = self.socket.recv(2048)
        print(str(receive, encoding='gbk'))
        self.socket.send(bytes('hi\n'))
        self.socket.close()

while True:
    connect_socket, client_addr = server.accept()
    TServer(connect_socket, client_addr).start()
    

# server main function
# thread = []
# thread.append(HandTrack('fdmdkw'))
# thread[0].start()
# thread[0].hand_track_model()
# thread[0].model_result()
