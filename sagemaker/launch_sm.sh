#!/usr/bin/env bash
#  # Default — no image rebuild, code uploaded from local source_dir:
# bash sagemaker/launch_sm.sh 1 sruthis sruthis cv-p5en p5en
# Force full image rebuild (when pip deps change):
# bash sagemaker/launch_sm.sh 1 sruthis sruthis cv-p5en p5en full

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
BUILD_TYPE=${6:-update}      # full / update
VERSION=${7:-210}
USER_NAME=sruthis

# Where the VLM model and data live *inside the container*
VLM_MODEL_PATH=${8:-"Qwen/Qwen2.5-VL-7B-Instruct"}
VLM_BASE_DATASET_PATH=${9:-"s3://tri-ml-sandbox-16011-us-west-2-datasets/sruthi_trl_training/na_na_16_expert_fulltask_PnPCounterToStove"}    # root with job dirs + eval_log + frames
VLM_SPLIT=${10:-"train"}                              # "train" or "val" or integer string
VLM_OUTPUT_DIR=${11:-"/opt/ml/model/vlm_overlay_outputs"}
batch_size_train=${12:-16}
batch_size_val=${13:-16}
BALANCE_DATA=${14:-true}
ENTRY_POINT=sagemaker/train_vlm_overlay_sm.py   # new tiny wrapper, see section 4
CONFIG=cosmos_predict2/configs/base/config.py   # PLACEHOLDER for any  config if needed

PROFILE=default
REGION=us-west-2
ARN=arn:aws:iam::124224456861:role/SageMaker-SageMakerAllAccess-us-west-2
S3_REMOTE_SYNC=s3://tri-ml-sandbox-16011-us-west-2-datasets/sagemaker/s3_remote_sync/
PRIORITY=${15:-100}

# ─── Training hyperparameters (edit these to change without rebuilding Docker) ───
EVAL_STRATEGY=${EVAL_STRATEGY:-steps}
LOGGING_STEPS=${LOGGING_STEPS:-500}
EVAL_STEPS=${EVAL_STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-500}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-500}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
REPORT_TO=${REPORT_TO:-wandb}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-sagemaker/ds_config_zero2.json}
BF16=${BF16:-true}
COMPARE_INTERVAL=${COMPARE_INTERVAL:-"4,8,12,16"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}

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
    --batch_size_train=${batch_size_train}              \
    --batch_size_val=${batch_size_val}                  \
    --vlm_model_name_or_path="${VLM_MODEL_PATH}"        \
    --vlm_base_dataset_path="${VLM_BASE_DATASET_PATH}"  \
    --vlm_output_dir="${VLM_OUTPUT_DIR}"                \
    --vlm_split="${VLM_SPLIT}"                          \
    --vlm_balance_data="${BALANCE_DATA}"                \
    --eval_strategy="${EVAL_STRATEGY}"                  \
    --logging_steps=${LOGGING_STEPS}                    \
    --eval_steps=${EVAL_STEPS}                          \
    --save_steps=${SAVE_STEPS}                          \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS}              \
    --learning_rate=${LEARNING_RATE}                    \
    --report_to="${REPORT_TO}"                          \
    --deepspeed_config="${DEEPSPEED_CONFIG}"            \
    --bf16="${BF16}"                                    \
    --compare_interval="${COMPARE_INTERVAL}"            \
    --gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
