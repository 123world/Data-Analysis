# YOLOV4 Deployment by Vitis AI Library 1.2
YOLO is the abbreviation of You Only Look Once, which is a state-of-the-art and real-time object detection system. YOLO algorithm series has vigorous vitality, therein YOLOV3 has been adopted by many algorithm engineers and now evolves to YOLOV4 which has optimal speed and accuracy. YOLOV4 uses __CSPDarknet53/SPP__ as backbone/plugin, it improves YOLOv3’s AP and FPS by 10% and 12% as follows figure:

<div align="center">
  <img width="60%" height="60%" src="doc/image/YOLOV4.png">
</div>

Xilinx Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with unified APIs, and it fully supports XRT 2020.1. The Vitis AI Library provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks, this simplifies the use of deep-learning neural networks and allows users to focus more on the development of their applications rather than the underlying hardware.
This article is a detailed introduction about how to deploy Darknet YOLOV4 model on Xilinx MPSOC platform by Vitis AI Library. The latest Vitis AI Library 1.2 is in Xilinx Vitis AI package which can be downloaded from [Vitis AI](https://github.com/Xilinx/Vitis-AI)

## Darknet YOLOV4 model selection and mAP test
Original YOLOV4 is trained in Darknet framework, user could get more information on [YOLOV4 repository](https://github.com/AlexeyAB/darknet). There are several YOLOV4 models with different structure in [YOLOV4 model zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo), we select YOLOv4-Leaky for deployment as Vitis AI 1.2 DPU does not support _Mish_ activation function. The detailed information for YOLOv4-Leaky model is as follows:

- **Backbone** - CSPDarknet53 with Leaky activation

- **Neck** - PANet with Leaky activation

- **Plugin** Modules – SPP (maxpool size 3,5,7)

- **BFLOPs** - 91@512x512

As Vitis AI 1.2 DPU constaint on maxpool kernel size(maximum is 8), the SPP part of YOLOV4 should be modified as:
    
    ### SPP ###
    [maxpool]
    stride=1
    size=3

    [route]
    layers=-2

    [maxpool]
    stride=1
    size=5

    [route]
    layers=-4

    [maxpool]
    stride=1
    size=7
    
    [route]
    layers=-1,-3,-5,-6
    ### End SPP ###

Darknet framework should be built for the YOLOV4 float model training/validation, with `GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=1` in the `Makefile` for acceleration. Please install `CUDA/CUDNN`in advance.
As the YOLOV4 is trained on COCO2014 dataset, user could use `./scripts/get_coco_dataset.sh` to get labeled MS COCO2014 detection dataset for retraining/validation. User could refer [YOLOV4 model training guide](https://github.com/AlexeyAB/darknet/wiki/Train-Detector-on-MS-COCO-(trainvalno5k-2014)-dataset) for model retraining. The command of YOLOV4 model validation is as follows:

    $ ./darknet detector valid cfg/coco.data cfg/yolov4-leaky.cfg yolov4-leaky.weights
_“cfg/coco.data”_ file is the configuration file which has following content:

    classes= 80
    train  = coco/trainvalno5k.txt
    valid = coco/5k.txt
    names = data/coco.names
    backup = backup/
    eval=coco

The validation result will be generated under “results/” folder as _“coco_results.json”_, user could use follows python evaluation code for mAP test.

    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    import skimage.io as io
    import pylab,json
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)

    def get_img_id(file_name):     
        ls = []     
        myset = []     
        annos = json.load(open(file_name, 'r'))     
        for anno in annos:       
            ls.append(anno['image_id'])     
        myset = {}.fromkeys(ls).keys()     
        return myset

    if __name__ == '__main__':    
        annType = ['segm', 'bbox', 'keypoints']#set iouType to 'segm', 'bbox' or 'keypoints'    
        annType = annType[1] # specify type here    
        cocoGt_file = 'coco/annotations/'instances_val2014.json'    
        cocoGt = COCO(cocoGt_file)    
        cocoDt_file = 'results/coco_results.json'    
        imgIds = get_img_id(cocoDt_file)
        cocoDt = cocoGt.loadRes(cocoDt_file)  
        imgIds = sorted(imgIds)    
        imgIds = imgIds[0:5000]    
        cocoEval = COCOeval(cocoGt, cocoDt, annType)     
        cocoEval.params.imgIds = imgIds    
        cocoEval.evaluate()    
        cocoEval.accumulate()    
        cocoEval.summarize()
Darknet YOLOv4-Leaky model mAP is :

<div align="center">
  <img width="90%" height="90%" src="doc/image/Darknet%20YOLOV4%20mAP.png">
</div>

## Caffe conversion and mAP test by caffe_xilinx

Darknet YOLOV4 should be converted to Caffe model for deployment on FPGA, Vitis AI 1.2 provide the caffe_xilinx framework under “Vitis-AI/AI-Model-Zoo” folder for Caffe conversion and inference, please install the `nccl/dependcy library` in advance and build up as following steps:

    $cp Makefile.config.example Makefile.config
    $make all
    $make pycaffe
    $export PYTHONPATH=~/Vitis-AI/AI-Model-Zoo/caffe_xilinx/python:$PYTHONPATH
    $python -c "import caffe; print caffe.__file__"//check the environment

After the caffe_xilinx environment building up, user could copy the darknet YOLOV4 model to current folder and run following script for conversion:

    python scripts/convert.py yolov4-leaky.cfg yolov4-leaky.weights yolov4-leaky.prototxt yolov4-leaky.caffemodel

Caffe YOLOV4 model will be generated after successful conversion, user could test this float model under caffe_xilinx environment using COCO2014 5K validation dataset. The result file in `.json` format can be generated by modifying the _yolo_detect.cpp_ under “examples/yolo/” folder. The detailed test script is as follows:

    ./build/examples/yolo/yolo_detect.bin yolov4-leaky.prototxt \
                                          yolov4-leaky.caffemodel \
                                          ./5K.txt    \
                                          -out_file yolov4_caffe_result.json \
                                          -confidence_threshold 0.005 \
                                          -classes 80 \
                                          -anchorCnt 3

Caffe YOLOv4-Leaky model mAP is：

<div align="center">
  <img width="90%" height="90%" src="doc/image/Caffe%20YOLOV4%20mAP.png">
</div>

## Quantization and compilation under docker environment
YOLOV4 model quantization and compilation need the installation of [Xilinx Vitis AI tools docker](https://github.com/Xilinx/Vitis-AI/blob/master/doc/install_docker/load_run_docker.md) on host server, loading the Vitis AI tools docker and entering the conda environment by following command:

    $ ./docker_run.sh xilinx/vitis-ai-gpu:latest
    /workspace$ conda activate vitis-ai-caffe
The input layer in yolov4-leaky.prototxt should be modified as follows for quantization:

     layer {
         name: "data"
         type: "ImageData"
         top: "data"
         top: "label"
         include {
             phase: TRAIN
         }
         transform_param {
             yolo_width: 512
             yolo_height: 512
         }
         image_data_param {
             source: "calib.txt"
             root_folder: ""
             batch_size: 10
             shuffle: false
         }
     }

Quantization script for Caffe YOLOV4 is as follows and the COCO2014 validation dataset could be used for calibration:

    vai_q_caffe quantize -model ./yolov4-leaky.prototxt \
                         -weights ./yolov4-leaky.caffemodel \
	                     -output_dir ./vai_q_output_yolov4_leaky_dpuv2 \
                         -calib_iter  200\
                         -method 1 \
                         -sigmoided_layers layer138-conv,layer149-conv,layer160-conv

There are 4 files will be generated under “vai_q_output_yolov4_leaky_dpuv2” folder after the successful quantization:

     quantize_train_test.prototxt
     quantize_train_test.caffemodel
     deploy.prototxt
     deploy.caffemodel
       

_quantize_train_test.*_ are for quantization result test and _deploy.*_ are for compilation.The input layer in _quantize_train_test.prototxt_ should be modified as same as the input layer in _deploy.prototxt_ for test, detailed script is as follows:

    ./build/examples/yolo/yolo_detect.bin quantize_train_test.prototxt \
                                          quantize_train_test.caffemodel \
                                          ./val2017.txt    \
                                          -out_file yolov4_caffe_result.json \
                                          -confidence_threshold 0.005 \
                                          -classes 80 \
                                          -anchorCnt 3

Quantized Caffe YOLOv4-Leaky model mAP is：

<div align="center">
  <img width="90%" height="90%" src="doc/image/Quantized%20YOLOV4%20mAP.png">
</div>

Compilation script for Caffe YOLOV4 is as follows and the ZCU102 is selected as deployment platform in this article:

    #!/bin/sh

    TARGET=ZCU102
    NET_NAME=yolov4
    DEPLOY_MODEL_PATH=vai_q_output_yolov4_leaky_dpuv2
    CF_NETWORK_PATH=.
    ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${TARGET}/${TARGET}.json

    vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.prototxt \
	            --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.caffemodel \
    	    --arch ${ARCH} \
    	    --output_dir ${CF_NETWORK_PATH}/vai_c_output_${TARGET}_dpuv2/ \
    	    --net_name ${NET_NAME} \
The _“dpu_yolov4.elf”_ file will be generated after the successful compilation which is for model running on the target platform.
## YOLOV4 model deployment by Vitis AI Library 1.2
### Step 1 Host setting up and Vitis AI Library building

User could modify and compile the deployment program based on Vitis AI library 1.2 source code on host server. Host cross-compilation system environment package of [sdk-2020.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.1.0.0.sh) should be downloadedand installed by following command:

    $./sdk-2020.1.0.0.sh   // “~/petalinux_sdk” path is recommended for the installation
    $unset LD_LIBRARY_PATH
    $source ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux

An extra package of [vitis_ai_2020.1-r1.2.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2020.1-r1.2.0.tar.gz) should be downloaded from 
and installed by following command:

    $tar -xzvf vitis_ai_2020.1-r1.2.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux

Vitis AI library 1.2 source code is based on C++. YOLOV4 has the same pre/post processing with YOLOV3, so user could reuse YOLOV3 framework source code. YOLOV3 pre/post processing source code respectively are _“yolov3/src/utils.cpp”_ and _“xnnpp/src/yolov3.cpp”_ which has been optimized for ARM processor, YOLOV3 deployment source code is _“yolov3/src/yolov3_imp.cpp”_, all the header dependency files are under “~/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/include/vitis/ai/” folder.
For the YOLOV3 framework AI library cross compilation, some parameters should be configured in CMakeLists.txt as follows:

    option(DISABLE_ALL "disable all libraries" ON)
    option(ENABLE_YOLOV3 "enable Yolov3" ON)

After following command execution, package of libvitis_ai_library-1.2.0-1.aarch64.rpm will be generated under default folder of “build/build.linux.2020.1.aarch64.Release/Vitis-AI-Library/”, it should be installed on target platform such as ZCU102 for YOLOV4 deployment.

    $./cmake.sh --clean --cmake-options='-DCMAKE_NO_SYSTEM_FROM_IMPORTED=on' --type=release --pack=rpm
Vitis AI library 1.2 also provides the test code for model deployment validation and mAP test, user could also reuse the YOLOV3 test code under “overview/samples/yolov3/” by following compilation command:

    $cd overview/samples/yolov3
    $bash -x build.sh

Several test programs will be generated for running on target platform. _test_accuracy_yolov3_voc.cpp_ can be reused and modified to generate the "result_yolov4.json" for YOLOv4 mAP test.
### Step 2 Target setting up and deployment

Vitis AI library 1.2 supports DPU integration flow both by Vitis and Vivado, please refer the [Vitis AI DPU TRD](https://github.com/Xilinx/Vitis-AI/tree/master/DPU-TRD/prj).

More information about Vitis DPU integration flow is as follows:

https://github.com/Xilinx/Vitis-AI/blob/master/DPU-TRD/prj/Vitis/README.md
https://github.com/Xilinx/Vitis_Embedded_Platform_Source/blob/2019.2/Xilinx_Official_Platforms/zcu102_dpu/release_notes.md

There are also some runtime libraries should be installed in order on target platform after DPU integration, user can download from [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library) and run following command:

    $tar -xzvf vitis-ai-runtime-1.2.0.tar.gz
    $scp -r vitis-ai-runtime-1.2.0/aarch64/centos root@IP_OF_BOARD:~/

    #cd centos
    #rpm -ivh --force libunilog-1.2.0-r<x>.aarch64.rpm
    #rpm -ivh --force libxir-1.2.0-r<x>.aarch64.rpm
    #rpm -ivh --force libtarget-factory-1.2.0-r<x>.aarch64.rpm
    #rpm -ivh --force libvart-1.2.0-r<x>.aarch64.rpm

Copy the libvitis_ai_library-1.2.0-1.aarch64.rpm that is generated in step 1 and install on the target platform.

    #rpm -ivh --force libvitis_ai_library-1.2.0-1.aarch64.rpm

User also need to create a “vitis_ai_library/models/yolov4” folder under “/usr/share/” on target board, and copy the _“dpu_yolov4.elf”_ file to this folder as _“yolov4.elf”_. A _“yolov4.prototxt”_ file with following content also need to be created in this folder.

    model {
      name: "yolov4"
      kernel {
         name: "yolov4"
         mean: 0.0
         mean: 0.0
         mean: 0.0
         scale: 0.00390625
         scale: 0.00390625
         scale: 0.00390625
      }
      model_type : YOLOv3
      yolo_v3_param {
        num_classes: 80
        anchorCnt: 3
        layer_name: "138"
        layer_name: "149"
        layer_name: "160"
        conf_threshold: 0.1
        nms_threshold: 0.45
        biases: 142
        biases: 110
        biases: 192
        biases: 243
        biases: 459
        biases: 401
        biases: 36
        biases: 75
        biases: 76
        biases: 55
        biases: 72
        biases: 146
        biases: 12
        biases: 16
        biases: 19
        biases: 36
        biases: 40
        biases: 28
        test_mAP: false
      }
    }

Copy the _test_jpeg_yolov3_ executable program and sample image from coco dataset to target board and run the command as follows:

    #./test_jpeg_yolov3 yolov4 sample_image.jpg

The result will be saved in samle_image_result.jpg:

<div align="center">
  <img width="60%" height="60%" src="doc/image/164_result.jpg">
</div>

Copy the _test_accuracy_yolov3_voc_ executable program and Coco2017 validation dataset to target board, modify the "conf_threshold" and "test_mAP" parameter in _“yolov4.prototxt”_ as :

    conf_threshold: 0.005
    test_mAP: true
 
Run the command as follows to get the test result on validation dataset:

    ./test_accuracy_yolov3_voc yolov4 val2017.txt result_yolov4.json
    
YOLOV4 deployment mAP is:

<div align="center">
  <img width="90%" height="90%" src="doc/image/Deployed%20YOLOV4%20mAP.png">
</div>

## References
- [Vitis AI Overview](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html)
- [Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_1/zkj1576857115470.html)
- [Vitis AI Model Zoo with Performance & Accuracy Data](https://github.com/Xilinx/AI-Model-Zoo)
- [Vitis AI Tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials)



