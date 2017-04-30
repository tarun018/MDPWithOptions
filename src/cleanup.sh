kill -9 `ps -ef | grep State.py | grep -v grep | awk '{print $2}'`
#rm ../Data/* rm *.run rm *.txt

