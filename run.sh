#!/bin/bash
###
 # @Author: Morgan Woods weiyiding0@gmail.com
 # @Date: 2025-03-09 17:41:04
 # @LastEditors: Morgan Woods weiyiding0@gmail.com
 # @LastEditTime: 2025-03-15 17:34:35
 # @FilePath: /SiriusxLLM/run.sh
 # @Description: 
###

/home/moguw/workspace/SiriusxLLM/build/demo/llama2 \
--checkpoint_path /home/moguw/workspace/SiriusxLLM/tmp/chat_q8.bin \
--tokenizer_path /home/moguw/workspace/SiriusxLLM/tmp/llama2.model \
--quantized true \
--prompt "How many pandas in the world" \
--use_cuda true # 如果没有CUDA就注释了