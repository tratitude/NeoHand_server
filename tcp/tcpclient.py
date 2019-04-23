from socket import socket, AF_INET, SOCK_STREAM
#import caffe

image = "C:\\Users\\Kellen\\NeoHand_server\\dataset\\picture\\webcam_0.png"
# image_full = caffe.io.load_image(image)
s = socket(AF_INET, SOCK_STREAM)
s.connect(('localhost', 20000))
with open(image, 'rb') as f:
    data = f.read()
s.send(data)