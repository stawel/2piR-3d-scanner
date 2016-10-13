
for((i=10000;i<11000;i++))
do
    echo $i
    read -n1 -r -p "Press space to continue..." key
    if [ "$key" == 'x' ]; then
	exit
    fi
    ./mkpic.sh $i
done