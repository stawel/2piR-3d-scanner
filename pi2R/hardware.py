import time
import RPi.GPIO as GPIO
import sys

stepper_pin_A = 2
stepper_pin_B = 3
stepper_pin_C = 4
stepper_pin_D = 14

stepper_position = 0
stepper_full_rotation=4096*3

stepper_steps = [1+2, 2+4, 4+8, 8+1]

def stepper_init():
    gpio_list = [stepper_pin_A,stepper_pin_B,stepper_pin_C,stepper_pin_D]
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(gpio_list, GPIO.OUT)
    stepper_position = 0


def _set_step(x):
#    print 'x:' ,x
    GPIO.output(A, x&1)
    GPIO.output(B, x&2)
    GPIO.output(C, x&4)
    GPIO.output(D, x&8)

def _make_step(x):
    x = x % 4;
    _set_step(stepper_steps[x])

def stepper_off():
    _set_step(0)

def stepper_goto(pos, direction = 1, delay = 0.03, turn_off = True):
    pos %= stepper_full_rotation
    while pos != stepper_position:
        stepper_position += direction
        stepper_position %= stepper_full_rotation
        _make_step(stepper_position)
        time.sleep(delay)
    if turn_off:
        stepper_off()

