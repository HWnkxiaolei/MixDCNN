# -*- coding: utf-8 -*-
"""
    Created on Wed Aug 19 15:25:47 2015
    
    @author: alexbewley
    
    @detail: This is a modified version of an earlier script used for generating a caffe network prototxt file.
    See Usage below and Zongyuan Ge's mixture DCNN paper for more details.
    
"""
import sys

if(len(sys.argv) < 4):
  print("Usage:\n$ python " + sys.argv[0] + " base-model.prototxt N K\n")
  print("Where:\n - base-model is the prototxt with special TAGS")
  print(" - N is the number of class outputs")
  print(" - K is the number of subset expert networks to replicate from the base model")
  exit()

base_model_definition = sys.argv[1]
num_outputs = int(sys.argv[2])
num_experts = int(sys.argv[3])

for s in range(num_experts):
  with open(base_model_definition,'r') as fin:
    for i,line in enumerate(fin):
      new_line = line.replace('EXPERT_NUM','se%d'%(s+1))
      new_line = new_line.replace('NUM_OUTPUTS',str(num_outputs))
      print(new_line),
    print(" ")

for s in range(num_experts):
  print("layers {\n  type:SPLICE\n  name:\"slice-fc8-se%d\"\n  bottom:\"fc8-se%d\""%(s+1,s+1))
  for i in range(num_outputs):
    print("  top: \"splice%d-%d\""%(s+1,i+1))
  print("}")
  
  print("layers {\n  type:ELTWISE\n  name:\"max-fc8-se%d\"\n  top:\"max-fc8-se%d\""%(s+1,s+1))
  for i in range(num_outputs):
    print("  bottom: \"splice%d-%d\""%(s+1,i+1))
  print("  eltwise_param{\n    operation:MAX\n  }\n}")


print("layers {\n  name: \"concat\"\n")
for s in range(num_experts):
  print("  bottom: \"max-fc8-se%d\""%(s+1))
print("  top: \"conf-ss\"\n  type: CONCAT\n  concat_param {\n    concat_dim: 1\n  }\n}")


print("layers {\n  name: \"prob-ss\"\n  type: SOFTMAX\n  bottom: \"conf-ss\"\n  top: \"prob-ss\"\n}")


print("layers {\n  name: \"slice-prob-ss\"\n  type: SLICE\n  bottom: \"prob-ss\"")
for s in range(num_experts):
  print("  top: \"prob-sw%d\""%(s+1))
print("}")


for s in range(num_experts):
  print("layers {")
  print("  name: \"repmat-sw%d\""%(s+1))
  print("  type: INNER_PRODUCT\n  bottom: \"prob-sw%d\""%(s+1))
  print("  top: \"repmat-sw%d\""%(s+1))
  print("  blobs_lr: 0\n  blobs_lr: 0")
  print("  weight_decay: 0\n  weight_decay: 0\n  inner_product_param {")
  print("    num_output: %d"%(num_outputs))
  print("    weight_filler {\n      type: \"constant\"\n      value: 1\n    }\n    bias_filler {\n      type: \"constant\"\n      value: 0\n    }\n  }\n}")

  print("layers {\n  type: ELTWISE")
  print("  name: \"weighted-prob-ss%d\""%(s+1))
  print("  bottom: \"fc8-se%d\""%(s+1))
  print("  bottom: \"repmat-sw%d\""%(s+1))
  print("  top: \"weighted-prob-ss%d\""%(s+1))
  print("  eltwise_param {\n    operation: PROD\n  }\n}")


print("layers {\n  name: \"sum-weighted-prob\"\n  type: ELTWISE")
for s in range(num_experts):
  print("  bottom: \"weighted-prob-ss%d\""%(s+1))
print("  top: \"prob\"")
print("  eltwise_param {\n    operation: SUM\n  }\n}")



#print loss layer
print("layers {\n" + \
"  bottom: \"prob\"\n" +\
"  bottom: \"label\"\n" + \
"  top: \"loss\"\n" + \
"  name: \"loss\"\n" + \
"  type: SOFTMAX_LOSS\n" +\
"  loss_weight: 1\n}")

#print accuracy layer
print("layers {\n" + \
"  name: \"accuracy\"\n" + \
"  type: ACCURACY\n" + \
"  bottom: \"prob\"\n" + \
"  bottom: \"label\"\n" + \
"  top: \"accuracy\"\n  include: { phase: TEST }\n}")




































