import socket
import cv2
import time
import sys
import struct
from PIL import Image
import numpy
import base64
from io import BytesIO

max_send_count  = 50
sleep_freq = 0.01
send_count = 0
recv_count = 0

host='140.127.208.181'
port=555
address=(host,port)
socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket.connect(address)

cap = cv2.VideoCapture(0)
tStart = time.time()#計時開始

while send_count < max_send_count:
    time.sleep(sleep_freq)
    ret, frame = cap.read()
    frame=cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)

    data=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    output_buffer = BytesIO()
    data.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    socket.send(base64_str)
    
    send_count = send_count+1
    message = 'Client: ' + str(socket.getsockname()) + ' ' + str(send_count)
    print(message)
    
    buf = socket.recv(65536)
    if not buf:
        break
    recv_count = recv_count + 1
    print('Server: ' + buf.decode('ascii'))

tEnd=time.time()
socket.close()
print('send fps:',(tEnd - tStart)/send_count)
