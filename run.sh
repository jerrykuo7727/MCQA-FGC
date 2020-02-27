#!/bin/bash

python3_cmd=python3.6

stage=0
use_gpu=cuda:0

model=bert
model_path=/home/M10815022/Models/bert-base-chinese
save_path=./models/ASR_text_only

train_datasets="ASR_train"
dev_datasets="ASR_dev"
test_datasets="ASR_test"
test_hard_datasets="ASR_test_hard"


if [ $stage -le 0 ]; then
  echo "==================================================="
  echo "     Convert traditional Chinese to simplified     "
  echo "==================================================="
  for dataset in $train_datasets $dev_datasets $test_datasets $test_hard_datasets; do
    file=dataset/$dataset.json
    echo "Converting '$file'..."
    opencc -i $file -o $file -c t2s.json || exit 1
  done
  echo "Done."
fi


if [ $stage -le 1 ]; then
  echo "======================"
  echo "     Prepare data     "
  echo "======================"

  rm -rf data
  for split in train dev test test_hard; do
    for dir in qcp1 qcp2 qcp3 qcp4 answer; do
      mkdir -p data/$split/$dir
    done
  done
  echo "Preparing dev set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path dev $dev_datasets || exit 1
  echo "Preparing test set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path test $test_datasets || exit 1
  echo "Preparing test-hard set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path test_hard $test_hard_datasets || exit 1
  echo "Preparing train set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path train $train_datasets || exit 1
fi

exit 0


if [ $stage -le 2 ]; then
  echo "================================="
  echo "     Train and test QA model     "
  echo "================================="
  #if [ -d $save_path ]; then
  #  echo "'$save_path' already exists! Please remove it and try again."; exit 1
  #fi
  #mkdir -p $save_path
  #$python3_cmd scripts/train_${model}.py $use_gpu $model_path $save_path
  #$python3_cmd scripts/finetune_${model}.py $use_gpu $model_path $save_path
  $python3_cmd scripts/validate_${model}.py $use_gpu $model_path $save_path
fi
