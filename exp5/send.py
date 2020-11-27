import socket


ip = "183.173.68.184"
'''This ip is your server machine's ip '''
port = 5000


def send_common(text):

    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((ip, port))

    c.send(bytes(text, encoding="utf-8"))
    rep = c.recv(1000)
    print(str(rep, encoding="utf-8"))
    return str(rep, encoding="utf-8")

    c.close()


if __name__ == "__main__":
    while True:
        cmd = "robot1"  # 假定机器人是robot1
        string = send_common(cmd)
        print("end")

