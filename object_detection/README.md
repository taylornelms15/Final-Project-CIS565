# TensorRT Objections Detection

# Overview

![](../images/detect.gif)



TensorRT is used to acclerate inference on the GPU. To use TensorRT one must first build the engine. This can be done in may ways using a UFF, Caffe Model or ONNX file. Because ONNX is so new and rapidly evolving I had a huge amount of version issues when attmepting to build a network from ONNX file. 

The easiest and least painful way is to freeze a tensorflow graph and then use nvidias uff converter to converter your tensorflow pb file to a uff file. If you can do this succesfully one huge hurdle is out of the way... Seriously it is painful...

The next step is to build the acclerated structure. One must parse the file and then set parameters accordingly.

# Performance Analysis

![](../images/trt_graph.png)

# Credits 

RandInt8Calibrator.cpp and plugin.cpp are from nvidia and are credited as such. They were needed to perform proper calibration and building of tensorRT engines
