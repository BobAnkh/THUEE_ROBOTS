import socket

# ip = "192.168.43.9"
ip = '192.168.43.9'
'''This ip is your server machine's ip '''
port = 12362


def send_common(text):

    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((ip, port))

    c.send(bytes(text, encoding="utf-8"))
    rep = c.recv(1000)
    print(str(rep,encoding="utf-8"))
    c.close()
    return str(rep, encoding="utf-8")


def get_data_frame(text):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((ip, port))
    c.send(bytes(text, encoding="utf-8"))
    rep = c.recv(1000)
    c.close()

    message = eval(rep)
    Location = message['Location']
    Barrier = message['Barrier']
    Target = message['Target']
    Start = message['Start']
    End = message['End']

    print('Location:', Location,'Barrier:', Barrier,'Target:', Target,'Start:', Start, 'End:',End)
    return Location, Barrier, Target, Start, End


if __name__ == "__main__":
    send_common('red')
    while True:
        cmd = "robot1"  # 假定机器人是robot1
        string = send_common(cmd)
        get_data_frame(cmd)
        print("end")
