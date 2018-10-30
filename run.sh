#! /bin/sh
cd `dirname $0`
libs="target/classes"
for f in target/lib/*
do
    libs=$libs:$f
done
java -classpath $libs perf.cfg.dl4jfacenet.FaceDetection $@
