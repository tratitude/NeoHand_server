import os

DIR = 'C:\\Users\\Kellen\\Pictures\\dataset\\webcam4\\'
width = 640
height = 480
bb_file = open(DIR+'boundbox.txt', 'w')
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