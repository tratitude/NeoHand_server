import cv2, os, time

def create_boundbox(DIR, width, height):
	#DIR = 'C:\\Users\\Kellen\\Pictures\\ganerate dataset\\webcam\\'
	
	bb_file = open(DIR+'\\boundbox.txt', 'w')
	file_num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
	file_num = file_num - 1
	
	# file containing one line per image with: u_start, v_start, u_end, v_end of hand
	# u -> width, v -> height
	if width > height:
		offset = int((width - height)/2)
		u_start = offset
		v_start = 1
		u_end = width - offset
		v_end = height
	else:
		offset = int((height - width)/2)
		u_start = 1
		v_start = offset
		u_end = width
		v_end = height - offset
	for i in range(file_num):
		bb_file.write('{:} {:} {:} {:}\n'.format(u_start, v_start, u_end, v_end))

	bb_file.close()
	
DIR = "C:\\Users\\Kellen\\Pictures\\dataset\\webcam5"
DIR_flip = "C:\\Users\\Kellen\\Pictures\\dataset\\webcam5_flip"
if not os.path.exists(DIR):
    os.makedirs(DIR)  
if not os.path.exists(DIR+"\\result"):
        os.makedirs(DIR+"\\result")
if not os.path.exists(DIR+"\\result_py"):
        os.makedirs(DIR+"\\result_py")

if not os.path.exists(DIR_flip):
    os.makedirs(DIR_flip)
if not os.path.exists(DIR_flip+"\\result"):
        os.makedirs(DIR_flip+"\\result")
if not os.path.exists(DIR_flip+"\\result_py"):
        os.makedirs(DIR_flip+"\\result_py")
        
num_frames = 300
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
cap.set(5,30)
#cap.set(10,5)
#cap.set(11,20)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cam_fps =cap.get(cv2.CAP_PROP_FPS)

print("cam_fps {0}".format(cam_fps))

i=1
x=-1
while(1):
    # get a frame
    ret, frame = cap.read()
    
    frame_flip=cv2.flip(frame,1)
    # show a frame
    cv2.imshow("capture", frame_flip)
    if x==1 and i <= num_frames:
        print(i)
        cv2.imwrite(DIR+"\\webcam_"+str(i)+".jpg", frame)
        cv2.imwrite(DIR_flip+"\\webcam_"+str(i)+".jpg", frame_flip)
        i=i+1
    elif  x==0:
        print ("Capturing {0} frames".format(num_frames))
        start = time.time()
        x = 1
    elif i==num_frames+1:
        end = time.time()
        seconds = end - start
        print ("Time taken : {0} seconds".format(seconds))
        fps  = num_frames / seconds;
        print ("Estimated frames per second : {0}".format(fps))
        break
    #輸入q開始 一定要英文
    if cv2.waitKey(1) & 0xFF == ord('v'):
        x=0
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #按q拍照 可能有延遲
    elif cv2.waitKey(1) & 0xFF == ord('p'):
        print(i)
        cv2.imwrite(DIR+"\\webcam_"+str(i)+".jpg", frame)
        cv2.imwrite(DIR_flip+"\\webcam_"+str(i)+".jpg", frame_flip)
        i=i+1

cap.release()
cv2.destroyAllWindows()
create_boundbox(DIR, 640, 480)
