#!/usr/bin/python

import sys
sys.path.append('../..')

from pi2R.hardware import *


stepper_init()
laser_init()

#stepper_goto(12)

laser(1)
x=2048*3
x/=4
#x/=8
stepper_goto(-x,-1)
#laser(0)
#stepper_goto(600,1)
#laser(1)
#stepper_goto(0,-1)
laser_off()


#stepper_goto(0,-1)

#stepper_goto(+1200,-1)

#stepper_goto(0)

cleanup()