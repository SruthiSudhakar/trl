#!/usr/bin/env bash
set -e

# NOTE: these unset are important; if not done AWS_PROFILE set in launcher will be ignored
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_PROFILE

INSTANCE_COUNT=$1     # e.g. 1
EXPERIMENT=$2        # free-form tag, or leave empty
NAME=$3               # short job name suffix

QUEUE_NAME=${4:-cv-p5en}
INSTANCE_TYPE=${5:-p5en}     # p4d, p4de, p5, p5en, g6e, ...
BUILD_TYPE=${6:-full}      # full / update
VERSION=${7:-210}
USER_NAME=sruthis

# Where the VLM model and data live *inside the container*
VLM_MODEL_PATH=${8:-"s3://tri-ml-sandbox-16011-us-west-2-datasets/sruthi_trl_training/Qwen2.5-VL-7B-Instruct"}
VLM_BASE_DATASET_PATH=${9:-"s3://tri-ml-sandbox-16011-us-west-2-datasets/sruthi_trl_training/na_na_16_expert_fulltask_PnPCounterToStove"}    # root with job dirs + eval_log + frames
VLM_SPLIT=${10:-"train"}                              # "train" or "val" or integer string
VLM_OUTPUT_DIR=${11:-"/opt/ml/model/vlm_overlay_outputs"}
batch_size_train=${12:-16}
batch_size_val=${13:-16}
ENTRY_POINT=sagemaker/train_vlm_overlay_sm.py   # new tiny wrapper, see section 4
CONFIG=cosmos_predict2/configs/base/config.py   # PLACEHOLDER for any  config if needed

PROFILE=default
REGION=us-west-2
ARN=arn:aws:iam::124224456861:role/SageMaker-SageMakerAllAccess-us-west-2
S3_REMOTE_SYNC=s3://tri-ml-sandbox-16011-us-west-2-datasets/sagemaker/s3_remote_sync/
PRIORITY=${12:-100}

AWS_DEFAULT_REGION=${REGION}                            \
    python3 sagemaker/launch_sagemaker_vlm.py    \
    --base-job-name=${USER_NAME}-vlm-overlay            \
    --entry_point=${ENTRY_POINT}                        \
    --user=${USER_NAME}                                 \
    --config=${CONFIG}                                  \
    --experiment=${EXPERIMENT}                          \
    --instance-count=${INSTANCE_COUNT}                  \
    --profile=${PROFILE}                                \
    --region=${REGION}                                  \
    --arn=${ARN}                                        \
    --s3-remote-sync=${S3_REMOTE_SYNC}                  \
    --priority=${PRIORITY}                              \
    --queue=${QUEUE_NAME}                               \
    --name=${NAME}                                      \
    --version=${VERSION}                                \
    --instance-type=${INSTANCE_TYPE}                    \
    --build-type=${BUILD_TYPE}                          \
    --version=${VERSION}                                \
    --batch_size_train=${batch_size_train}              \
    --batch_size_val=${batch_size_val}                  \
    --vlm_model_name_or_path="${VLM_MODEL_PATH}"        \
    --vlm_base_dataset_path="${VLM_BASE_DATASET_PATH}"  \
    --vlm_output_dir="${VLM_OUTPUT_DIR}"                \
    --vlm_split="${VLM_SPLIT}"

