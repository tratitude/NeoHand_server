# -*- coding: utf-8 -*-
from model.EvalRegNet_ImageSequence import HandTrack
import multiprocessing as mp
import cv2
import time
import os
from PIL import Image
import numpy as np

bb_offset = 25
test_iteration = 10
model_freq = 1
max_frame = 1000

if __name__=='__main__':
	os.environ['GLOG_minloglevel'] = '2'
	
	img_list = []
	avg_input_fps = []
	avg_output_fps = []
	
	cap = cv2.VideoCapture(0)
	cap.set(3,640)
	cap.set(4,480)
	# fps setting, but useless on VX-1000 webcam
	cap.set(5, 30)

	test_result_file = open('test_performance.txt', 'w')
	test_result_file.write('Total iterations: {}\n'.format(test_iteration))
	test_result_file.close()

	for i in range(test_iteration):
		#print('Iteration: {} start'.format(i))
		recv_que = mp.Queue()
		send_que = mp.Queue()
		ht = HandTrack('fdmdkw', model_freq, bb_offset, recv_que, send_que)
		ht.start()

		frame_counter = 0
		img_list.clear()

		# timer start
		tStart = time.time()
		while(frame_counter < max_frame):
			#print('frame :{}'.format(frame_counter))
			ret, frame = cap.read()
			
			#cv2.imshow('frame', frame)
			
			img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
			img = np.array(img).astype('int32')
			img_list.append(img)

			frame_counter = frame_counter + 1

			if frame_counter % model_freq == 0:
				inputbuf = img_list[:]
				recv_que.put(inputbuf)
				img_list.clear()
			
			
		# timer stop
		tStop = time.time()
		ht.terminate()

		test_result_file = open('test_performance.txt', 'a')
		test_result_file.write('Iteration: {}\n'.format(i))
		input_fps = frame_counter/(tStop-tStart)
		test_result_file.write('Input frames: {}\t time: {}\t fps: {}\n'.format(frame_counter, tStop-tStart, input_fps))
		output_fps = send_que.qsize()/(tStop-tStart)
		test_result_file.write('Output frames: {}\t time: {}\t fps: {}\n'.format(send_que.qsize(), tStop-tStart, output_fps))
		test_result_file.close()

		avg_input_fps.append(input_fps)
		avg_output_fps.append(output_fps)

		if ht.is_alive:
			ht.join()
		#print('Iteration: {} finished'.format(i))
	cap.release()

	test_result_file = open('test_performance.txt', 'a')
	input_fps = sum(avg_input_fps) / len(avg_input_fps)
	output_fps = sum(avg_output_fps) / len(avg_output_fps)
	test_result_file.write('Average input fps: {}\t output fps:{}\n\n'.format(input_fps, output_fps))
	test_result_file.close()