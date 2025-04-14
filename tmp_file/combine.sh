#!/bin/bash
# 首先加载宿主机A的环境变量
while IFS='=' read -r name value; do
    export "$name"="$value"
done < hostA_env.txt

# 然后加载宿主机B的环境变量，如果有重复的变量名，则覆盖宿主机A的设置
while IFS='=' read -r name value; do
    export "$name"="$value"
done < hostB_env.txt
