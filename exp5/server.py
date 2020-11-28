import socket
import time

ip = "192.168.43.174"
post = 5000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, post))
s.listen(10)
loc = [11, 24, 32, 47, 58, 69, 70, 18]


def get_common():
    while True:
        conn, addr = s.accept()
        print(addr)
        while True:

            try:
            #if True :
                data = conn.recv(1000)

                if data == b"byee":
                    break
                data = str(data, encoding="utf-8")
                #print(data2)
                if data[:5] == "robot" :
                    print("new")
                    num = int(data[5])
                    #print(num)
                    # location = str(loc[num])
                    print('Current location request: ')
                    location = str(input())
                    print(location)
                    #conn.send(bytes(location+","+location, encoding="utf-8"))
                    conn.send(bytes(location, encoding="utf-8"))
                    print("sucess?")
                else:
                    conn.send(bytes("data", encoding="utf-8"))
                print(data)
        #data=str(data[:4])

            except :
                print("pending")
                break


        conn.close()
    s.close()


if __name__ == "__main__":
    get_common()

