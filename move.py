#!/usr/bin/python
import time
from pi2R.hardware import *
import tty
import sys
import termios




stepper_init()
laser_init()

pos = 0
pos_step=16


def move(step):
    global pos
    pos+=step
    print("move pos:", pos)
    d = 1
    if step < 0: d = -1
    stepper_goto(pos,d, delay=0.010, turn_off=False)



orig_settings = termios.tcgetattr(sys.stdin)

tty.setcbreak(sys.stdin)
x = 0
print("press 'a' or 's'")
while x != chr(27): # ESC
    x=sys.stdin.read(1)[0]
    if x == 'a': move(-pos_step)
    if x == 's': move(pos_step)
    if x == 'z': move(-1)
    if x == 'x': move(1)
#    print("You pressed", x)
    termios.tcflush(sys.stdin, termios.TCIOFLUSH)

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)


stepper_off()