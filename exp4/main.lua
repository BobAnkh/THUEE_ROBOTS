dofile('0:/lua/lib.lua')

while(true)
do
  A = HKEY()
  if A == 193 then


    -- 伸出支撑手
    MOTOsetspeed(30)
    MOTOmove19(80,27,139,100,93,55,124,100,120,172,60,100,107,145,76,100,128,71,100)
    MOTOwait()


    -- 扑街
    MOTOsetspeed(12)
    MOTOmove19(80,27,139,100,156,125,69,100,120,173,60,100,51,77,134,100,127,71,100)
    MOTOwait()
    DelayMs(100)


    -- 放腿
    MOTOsetspeed(30)
    MOTOmove19(80,23,186,100,157,10,110,100,120,173,17,100,44,190,84,103,127,71,100)
    MOTOwait()


    -- 向前移
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,120,170,100,100,107,145,76,100,128,71,100)
    MOTOwait()


    -- 放腿
    MOTOsetspeed(30)
    MOTOmove19(80,23,186,100,157,10,110,100,120,173,17,100,44,190,84,103,127,71,100)
    MOTOwait()


    -- 向后伸手
    MOTOsetspeed(30)
    MOTOmove19(80,23,105,100,157,10,110,100,120,172,84,100,44,190,84,103,127,71,100)
    MOTOwait()


    -- 伸展双臂
    MOTOsetspeed(30)
    MOTOmove19(82,88,153,100,157,10,110,100,120,110,42,100,44,190,84,103,127,71,100)
    MOTOwait()


    -- 旋转双臂
    MOTOsetspeed(30)
    MOTOmove19(82,87,190,100,157,10,110,100,120,111,10,100,44,190,84,103,127,71,100)
    MOTOwait()
    MOTOrigid16(40,100,100,40,40,40,40,40,40,100,100,40,40,40,40,40)


    -- 收双臂1
    MOTOsetspeed(30)
    MOTOmove19(28,66,186,100,157,10,110,100,166,124,10,100,44,190,84,103,127,71,100)
    MOTOwait()
    MOTOrigid16(40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40)


    -- 劈腿
    MOTOsetspeed(30)
    MOTOmove19(28,66,186,52,158,113,48,95,166,124,10,160,46,79,156,99,127,71,100)
    MOTOwait()


    -- 撑起来
    MOTOsetspeed(30)
    MOTOmove19(75,37,190,52,158,113,48,83,113,175,10,160,46,79,156,117,128,71,100)
    MOTOwait()


    -- 叉开腿
    MOTOsetspeed(30)
    MOTOmove19(79,23,173,47,155,148,47,61,120,185,31,148,46,53,145,140,127,71,100)
    MOTOwait()
    DelayMs(100)


    -- 双臂向后
    MOTOsetspeed(30)
    MOTOmove19(79,120,173,47,155,148,47,61,120,81,10,148,46,53,145,140,127,71,100)
    MOTOwait()
    MOTOrigid16(40,40,40,100,100,40,40,40,40,40,40,100,100,40,40,40)


    -- 收脚
    MOTOsetspeed(30)
    MOTOmove19(79,120,173,88,110,148,47,81,120,81,10,114,78,53,145,118,127,71,100)
    MOTOwait()
    MOTOrigid16(40,40,40,100,100,40,40,100,40,40,40,100,100,40,40,100)


    -- 再收脚
    MOTOsetspeed(30)
    MOTOmove19(79,120,173,104,110,148,47,92,120,81,10,102,78,53,145,112,127,71,100)
    MOTOwait()


    -- 调整手臂
    MOTOsetspeed(30)
    MOTOmove19(79,27,64,104,110,148,47,92,120,172,140,102,78,53,145,112,127,71,100)
    MOTOwait()


    -- 站起来
    MOTOsetspeed(30)
    MOTOmove19(80,25,91,100,93,55,124,100,120,172,114,100,107,145,76,100,128,71,100)
    MOTOwait()


    -- 直立
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,120,170,100,100,107,145,76,100,128,71,100)
    MOTOwait()

  HKEY()
  end
  if A == 196 then


    -- 伸展右臂
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,126,108,10,100,107,145,76,100,128,71,100)
    MOTOwait()


    -- 右臂向前
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,120,187,10,100,107,145,76,100,128,40,100)
    MOTOwait()

  HKEY()
  end
  if A == 195 then


    -- 伸出右臂
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,119,188,10,100,107,145,76,100,128,40,100)
    MOTOwait()
    DelayMs(100)


    -- 放下右臂
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,119,179,48,100,107,145,76,100,128,40,100)
    MOTOwait()
    DelayMs(100)
    MOTOrigid16(40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40)


    -- 加紧右手指
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,121,179,48,100,107,145,76,100,128,63,100)
    MOTOwait()
    DelayMs(200)


    -- 抬起右臂
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,121,187,10,100,107,145,76,100,128,63,100)
    MOTOwait()
    DelayMs(400)

  HKEY()
  end
  if A == 194 then


    -- 抬起右臂
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,121,187,10,100,107,145,76,100,128,63,100)
    MOTOwait()
    DelayMs(400)


    -- 右臂不旋转
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,121,108,10,100,107,145,76,100,128,63,100)
    MOTOwait()


    -- 右臂向右伸展
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,117,92,96,100,107,145,76,100,128,61,100)
    MOTOwait()
    DelayMs(200)


    -- 右臂置于右侧
    MOTOsetspeed(30)
    MOTOmove19(80,30,100,100,93,55,124,100,117,178,96,100,107,145,76,100,128,61,100)
    MOTOwait()
    DelayMs(200)

  HKEY()
  end

HKEY()
end