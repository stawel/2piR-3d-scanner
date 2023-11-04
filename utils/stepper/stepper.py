#!/usr/bin/python
import time
import RPi.GPIO as GPIO
import sys

A = 2
B = 3
C = 4
D = 14

gpio_list = [A,B,C,D]
GPIO.setmode(GPIO.BCM)
GPIO.setup(gpio_list, GPIO.OUT)


def setStep(x):
#    print ('x:' ,x)
    GPIO.output(A, x&1)
    GPIO.output(B, x&2)
    GPIO.output(C, x&4)
    GPIO.output(D, x&8)

v = [1+2, 2+4, 4+8, 8+1]

def makeStep(x):
    x = x % 4;
    setStep(v[x])


for i in range(0,2560):
    print ('step: ',i)
    makeStep(-i)
    time.sleep(0.10)
    setStep(0)
    sys.stdin.readline()
    



setStep(0)