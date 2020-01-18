#!/bin/sh

set -e

export TF_CPP_MIN_LOG_LEVEL=3

python3 export.py

echo "> Freezing the computational graph for better performance"
python3 -m tensorflow.python.tools.freeze_graph --input_graph=models/model.pb \
  	--input_checkpoint='models/checkpoint-0' \
  	--output_graph=models/frozen.pb \
  	--output_node_names='pred_score_map/Sigmoid,pred_geo_map/concat'


echo "> Optimizing the model for inference"
python3 -m tensorflow.python.tools.optimize_for_inference --input=models/frozen.pb \
	--output=models/optimized.pb \
	--input_names=input_image \
	--output_names='pred_score_map/Sigmoid,pred_geo_map/concat'

# echo "> Generating the onnx model"
# python3 -m tf2onnx.convert --input models/optimized.pb --inputs "input_image:0[1,224,224,3]" --outputs "pred_score_map/Sigmoid:0,pred_geo_map/concat:0" --output models/model.onnx --opset 11 --verbose 
