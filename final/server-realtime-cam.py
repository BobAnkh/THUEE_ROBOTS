from time import sleep
import socket
import time
import threading
import _thread
import serial
import math
import json

length = 30
count = 0

# ip = '192.168.43.8'
ip = '127.0.0.1'
post = 5000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, post))
s.listen(10)

loc = [11, 24, 32, 47, 58, 69, 70, 18]
dis = "0"
ang = "0"
x = 0
y = 0
location = "default"


class Thread2 (threading.Thread):
    def __init__(self, threadID, name, counter):
        super(Thread2, self).__init__()
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def get_common(self):

        conn, addr = s.accept()
        print(addr)

        # try:
        # if True:
        data = conn.recv(1000)

        data = str(data, encoding="utf-8")
        # print(data2)
        if data[:5] == "robot":
            print("new")
            num = int(data[5])
            # print(num)

            global x, y

            conn.send(bytes(x + "," + y+","+str(count), encoding="utf-8"))
            print("sucess?")
        else:
            print("wrong")
        print(data)
        # data=str(data[:4])

        # except :
        # print("pending")
        # break

        conn.close()


def get_common():
    while True:
        print("waiting for connection ")
        conn, addr = s.accept()
        print(addr)
        if True:
            print("transmitting")
            # try:
            if True:
                data = conn.recv(1000)

                if data == b"byee":
                    break
                data = str(data, encoding="utf-8")
                # print(data2)
                if data[:5] == "robot":
                    # print("new")
                    num = int(data[5])
                    # print(num)
                    global location
                    input = {"Location": [x, y], "Barrier": [
                        0, 0], "Target": 0, "Start": [0, 0], "End": [0, 0]}
                    out = json.dumps(input)
                    conn.send(bytes(out, encoding="utf-8"))
                    # location=str(loc[num])
                    # print(location)
                    # conn.send(bytes(str(x)+","+str(y)+","+str(count), encoding="utf-8"))
                    # print("sucess?")
                else:
                    conn.send(bytes(dis+","+ang, encoding="utf-8"))
                    break
                print(data)
        # data=str(data[:4])

            # except :
                # print("pending")
                # break

        conn.close()
    s.close()


def set():
    global x, y
    # notice here , type in two numbers with out space between them .
    x1 = (input("enter x:  "))
    # if you want to update location to 1,2 , type in 11 and press enter .
    threadLock.acquire()

    x = x1[0]
    if len(x1) > 1:
        y = x1[1]
    threadLock.release()
    pass


threadLock = threading.Lock()

if __name__ == "__main__":
    xx = 0
    yy = 15
    # /dev/ttyUSB0
    while True:

        # xx+=1
        # yy+=1
        # x = (input("enter x:  "))

        # y = (input("enter y:  "))
        print("location set ! , waiting fot requests loaction= ", x, y)
        t = threading.Thread(target=get_common, args=())
        t2 = threading.Thread(target=set, args=())
        t.start()
        t2.start()
        count = 0

        t.join(3)
        t2.join(3)
        print("end")
        # get_common()
