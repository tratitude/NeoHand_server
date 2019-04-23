from socketserver import BaseRequestHandler, TCPServer
#from ..model.EvalRegNet_ImageSequence import HandTrack

image = "C:\\Users\\Kellen\\NeoHand_server\\dataset\\picture\\webcam_0_recv.png"


class EchoHandler(BaseRequestHandler):
    def handle(self):
        print('Got connection from', self.client_address)
        while True:
            msg = self.request.recv(8192)
            if not msg:
                break
            #self.request.send(msg)
            with open(image, 'wb') as f:
                f.write(msg)


if __name__ == '__main__':
    serv = TCPServer(('', 20000), EchoHandler)
    serv.serve_forever()
    '''
    thread = []
    thread.append(HandTrack('fdmdkw'))
    thread[0].start()
    thread[0].hand_track_model()
    thread[0].model_result()
    '''
