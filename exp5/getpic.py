from PIL import Image as PIL_Image
import os

obj_type = 'null'
path = '/home/xilinx/jupyter_notebooks/photo/data/'+obj_type
if ~os.path.exists(path):
    os.makedirs(path)

for i in range(0, 30):
    img_path = path+'/'+str(i)+'.jpg'
    !fswebcam  --no-banner --no-overlay --save {img_path} -d /dev/video2 2> /dev/null
    print(i)
    if (i+1) % 25 == 0:
        input()
