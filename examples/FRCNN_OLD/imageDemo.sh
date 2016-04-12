# Determine platform
echo Operating System: $UNAME

../../build/examples/FRCNN_OLD/Faster_Test.bin test.jpg result.jpg .config "-1" 2>&1|tee detection.log
