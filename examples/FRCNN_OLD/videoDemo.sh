# Determine platform
UNAME=$(uname -s)
echo Operating System: $UNAME

../../build/examples/FRCNN_OLD/Faster_Demo.bin .config "-1" 2>&1|tee videoDemo.log
