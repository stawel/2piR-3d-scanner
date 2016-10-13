
OUTPUT=$1

#echo 18 > /sys/class/gpio/export
#echo out > /sys/class/gpio/gpio18/direction

echo 0 > /sys/class/gpio/gpio18/value
raspistill -o ${OUTPUT}_1.jpg
echo 1 > /sys/class/gpio/gpio18/value
raspistill -o ${OUTPUT}_2.jpg
#echo 0 > /sys/class/gpio/gpio18/value
