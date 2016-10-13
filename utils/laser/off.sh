
OUTPUT=$1

echo 18 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio18/direction

echo 0 > /sys/class/gpio/gpio18/value
