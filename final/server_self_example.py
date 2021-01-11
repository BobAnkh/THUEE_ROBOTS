# Run InPython 3
# Change Below Line 47 and Line 61 to initial
# Send 'red'/'blue'
# then send whatever to receive all
# every time starting might need to change port in line 84
# type XrYrXbYb(4 numbers without space between each other) in console to change realtime position
from socketserver import BaseRequestHandler, ThreadingTCPServer
import threading
import atexit
red = False
blue = False
globalRedX = 0
globalRedY = 0
globalBlueX = 0
globalBlueY = 0


def set():
  global globalRedX,globalRedY,globalBlueX,globalBlueY
  while(1):
    x1 = (input("enter x:  "))

    if len(x1)==4 :
      globalRedX = int(x1[0])
      globalRedY = int(x1[1])
      globalBlueX= int(x1[2])
      globalBlueY= int(x1[3])
      print(x1)


class EchoHandler(BaseRequestHandler):
  def handle(self):
    global red
    global blue

    global globalRedX,globalRedY,globalBlueX,globalBlueY
    print('Got connection from', self.client_address)
    msg = self.request.recv(8192)
    msg = msg.decode(encoding="utf-8")
    if not red:
      if "red" in msg:
        red = self.client_address[0]
        self.request.send(b"ok")
        self.request.close()
        return
    if not blue:
      if "blue" in msg:
        blue=self.client_address[0]
        self.request.send(b"ok")
        self.request.close()
        return
    print((red,blue))
    if(red == self.client_address[0]):
      ## change here
      (xStart,yStart)=(1,1)#redStartPoint
      (xEnd,yEnd)=(1,1)#redEndPoint
      (Tar1,Tar2) = (2,3)#redPosibleTargetPositions
      barrierInner = 1
      barrierOuter = 1
      self.request.send(b'{"Location":[%d,%d],"Barrier":[%d,%d],"Target":[%d,%d],"Start":[%d,%d],"End":[%d,%d],"Type":"red"}'%(
        globalRedX,globalRedY,
        barrierInner,barrierOuter,#barrierInner,barrierOuter,
        Tar1,Tar2,
        round(xStart),round(yStart),xEnd,yEnd
      ))
      self.request.close()
      return 
    if(blue == self.client_address[0]):

      ## change here
      (xStart,yStart)=(1,1)#redStartPoint
      (xEnd,yEnd)=(1,1)#redEndPoint
      (Tar1,Tar2) = (2,3)#redPosibleTargetPositions
      barrierInner = 1
      barrierOuter = 1
      self.request.send(b'{"Location":[%d,%d],"Barrier":[%d,%d],"Target":[%d,%d],"Start":[%d,%d],"End":[%d,%d],"Type":"blue"}'%(
        globalBlueX,globalBlueY,
        barrierInner,barrierOuter,
        Tar1,Tar2,
        round(xStart),round(yStart),xEnd,yEnd
      ))

      self.request.close()
      return
    self.request.send(b'vot valid')
    self.request.close()


if __name__ == '__main__':
  t2 = threading.Thread(target= set , args=())
  t2.start()
  ThreadingTCPServer.allow_reuse_address = True
  serv = ThreadingTCPServer(('', 12362), EchoHandler,)
  serv.allow_reuse_address = True
  def foo():
    print("exit")
    serv.shutdown()
  atexit.register(foo)
  serv.serve_forever()