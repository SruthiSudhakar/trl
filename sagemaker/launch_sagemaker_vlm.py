# custom/sagemaker/launch_sagemaker_vlm.py
import argparse
import time
import os
import subprocess
from datetime import datetime
from pathlib import Path

import boto3
from sagemaker import Session as sm_Session
from sagemaker.pytorch import PyTorch

try:
    from sagemaker.batch_queueing.queue import Queue
    IS_SM_QUEUE = True
except Exception as e:
    print(f"Could not load SageMaker batch queueing: {e}.")
    IS_SM_QUEUE = False

NAME = "vlm-overlay"
INSTANCE_MAPPER = {
    "p4d": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    "g6e": "ml.g6e.48xlarge",
    "g6e-small": "ml.g6e.12xlarge",
    "g5": "ml.g5.48xlarge",
    "g5-small": "ml.g5.24xlarge",
}


def run_command(command: str) -> None:
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(
    user: str,
    instance_type: str,
    version: str = "271",
    build_type: str = "full",
    profile: str = "default",
    region: str = "us-west-2",
) -> str:
    """
    Build or update the VLM-specific image.
    Uses Dockerfile_vlm_<version> and Dockerfile_vlm_update.
    """
    print(f"Building VLM image for user {user}, instance_type {instance_type}, version {version}, build_type {build_type}")
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    print(f"Docker directory: {docker_dir}")
    if instance_type in INSTANCE_MAPPER.keys():
        algorithm_name = f"{user}-{NAME}-{version}"
        dockerfile_base = docker_dir / f"Dockerfile_{version}"
        dockerfile_update = docker_dir / "Dockerfile_update"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")

    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    if build_type is None:
        return fullname

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"

    if build_type == "full":
        commands = [
            f"{login_cmd} 124224456861.dkr.ecr.{region}.amazonaws.com",
            f"{login_cmd} {registry}",
            f"docker build -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            (
                f"aws --region {region} --profile {profile} ecr describe-repositories --repository-names {algorithm_name} || "
                f"aws --region {region} --profile {profile} ecr create-repository --repository-name {algorithm_name}"
            ),
        ]
    elif build_type == "update":
        base_image = fullname
        commands = [
            f"docker pull {base_image}",
            f"docker build -f {dockerfile_update} --build-arg BASE_DOCKER={base_image} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]
    else:
        raise ValueError(f"Unknown build_type: {build_type}")

    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return fullname


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", default="full")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")
    parser.add_argument("--entry_point", type=str, default="scripts/train_vlm_overlay.py")

    # Keep config arg to mirror your existing scripts (unused here)
    parser.add_argument("--config", help="config base", required=True)
    parser.add_argument("--experiment", help="which experiment to run", type=str, default=None)

    # AWS / SM
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="SageMaker execution role ARN")
    parser.add_argument("--s3-remote-sync", default=None, help="S3 root to sync outputs")

    # Instances
    parser.add_argument("--instance-count", default=1, type=int)
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--version", default="271", type=str)
    parser.add_argument("--spot-instance", action="store_true")

    # Job naming / queue
    parser.add_argument("--base-job-name", type=str)
    parser.add_argument("--input-source", choices=["s3", "lustre", "local"], default="s3")
    parser.add_argument("--fss-identifier", default="default")
    parser.add_argument("--priority", default=5, type=int)
    parser.add_argument("--queue", type=str, default="ml")
    parser.add_argument("--name", type=str, default=None)

    # VLM overlay specific
    parser.add_argument("--vlm_model_name_or_path", type=str, required=True)
    parser.add_argument("--vlm_base_dataset_path", type=str, required=True)
    parser.add_argument("--vlm_output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--vlm_split", type=str, default="train")
    parser.add_argument("--batch_size_train", type=int, default=16)
    parser.add_argument("--batch_size_val", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    # Map your short queue names to FSS queues (same as your existing pattern)
    if args.queue == "mvdp":
        args.queue = "fss-mvdp-p4de-24xlarge-us-west-2"
        args.instance_type = "p4de"
    elif args.queue == "vla":
        args.queue = "fss-vla-p4de-24xlarge-us-west-2"
        args.instance_type = "p4de"
    elif args.queue == "dp":
        args.queue = "fss-dp-p4de-24xlarge-us-west-2"
        args.instance_type = "p4de"
    elif args.queue == "ml":
        args.queue = f"fss-{INSTANCE_MAPPER[args.instance_type]}-{args.region}".replace(".", "-")
        args.instance_type = "p4de"
    elif args.queue == "mlp5":
        args.queue = "fss-ml-p5-48xlarge-us-west-2"
        args.instance_type = "p5"
    elif args.queue == "testing":
        args.queue = "fss-testing-p5-48xlarge-us-west-2"
        args.instance_type = "p5"
    elif args.queue == "cv-p5en":
        args.queue = "fss-cv-ml-p5en-48xlarge-us-west-2"
        args.instance_type = "p5en"
    else:
        raise ValueError(f"Invalid queue name {args.queue}")

    main_after_setup(args)


def main_after_setup(args):
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set SAGEMAKER_ARN env var"
        args.arn = os.environ["SAGEMAKER_ARN"]

    if args.s3_remote_sync is None:
        assert "S3_REMOTE_SYNC" in os.environ, "Please specify --s3-remote-sync or set S3_REMOTE_SYNC env var"
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]

    image_uri = get_image(
        args.user,
        args.instance_type,
        args.version,
        region=args.region,
        build_type=args.build_type,
        profile=args.profile,
    )

    # g-series do not use queue
    use_sm_queue = IS_SM_QUEUE and not args.instance_type.startswith("g")

    sagemaker_session = sm_Session(
        boto_session=boto3.session.Session(region_name=args.region, profile_name=args.profile)
    )

    if args.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()

    role = args.arn
    # account = boto3.client("sts").get_caller_identity()["Account"]
    account = "124224456861"
    session = boto3.session.Session(region_name=args.region)
    region = session.region_name

    base_job_name = args.base_job_name

    def get_job_name(base):
        now = datetime.now()
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
        return "-".join([base, date_str])

    if args.name is None:
        job_name = get_job_name(base_job_name)
    else:
        job_name = f"{base_job_name}--{args.name}".replace("_", "-")

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    tags = [
        {"Key": "tri.project", "Value": "MM:PJ-0077"},
        {"Key": "tri.owner.email", "Value": "fangzhou.cheng.ctr@tri.global"},
    ]

    max_run = 15 * 24 * 60 * 60
    max_wait = max_run if args.spot_instance else None
    keep_alive_period_in_seconds = 300 if not args.spot_instance else None
    entry_point = args.entry_point
    print(f"Using entry point script: {entry_point}")
    instance_type = "local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type]
    instance_count = args.instance_count
    train_use_spot_instances = args.spot_instance

    # Hyperparameters passed directly to sft_vlm_overlay_regression_v2.py
    hyperparameters = {
        "model_name_or_path": args.vlm_model_name_or_path,
        "output_dir": args.vlm_output_dir,
        "base_dataset_path": args.vlm_base_dataset_path,
        "split": args.vlm_split,
        "eval_strategy": "steps",
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 500,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": args.batch_size_train,
        "per_device_eval_batch_size": args.batch_size_val,
        "report_to": "wandb",
        "compare_interval": "4,8,12,16",
    }

    distribution = {
        "torch_distributed": {
            "enabled": True,
        }
    }

    environment = {
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "465628e1cdd752aed296abe6439dedcec6fb3292"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
        "WANDB__SERVICE_WAIT": "300",
        "HF_TOKEN": os.environ.get("HF_TOKEN", "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"),
        "HF_HOME": "/tmp",
        "INSTANCE_COUNT": str(args.instance_count),
        "SM_USE_RESERVED_CAPACITY": "1",
        "FI_EFA_FORK_SAFE": "1",
        "NVTE_FUSED_ATTN": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "AWS_S3_USE_CRT": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "SAGEMAKER_PROGRAM": entry_point,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "SM_JOB_NAME": job_name,
        "SAGEMAKER": "enabled",
        "QUEUE": str(args.queue),
        "INSTANCE_TYPE": str(args.instance_type),
        "INSTANCE_COUNT": str(args.instance_count),
        "NCCL_DEBUG": "TRACE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_DEBUG_SUBSYS": "ALL",
        "NCCL_DEBUG_FILE": "/opt/ml/output/nccl_%h_%p.log",
    }

    if args.name is not None:
        environment["JOB_NAME"] = args.name

    checkpoint_s3_uri = None if args.local else os.path.join(
        f"s3://tri-ml-sandbox-16011-us-west-2-datasets/sagemaker/{args.user}/{NAME}", job_name
    )
    checkpoint_local_path = None if args.local else "/opt/ml/checkpoints"

    print()
    print("#############################################################")
    print(f"SageMaker Execution Role:       {role}")
    print(f"SM Queue:                       {use_sm_queue}-{args.priority}-{args.fss_identifier}")
    print(f"AWS region:                     {region}")
    print(f"AWS profile:                    {args.profile}")
    print(f"AWS account:                    {account}")
    print(f"Entry point:                    {entry_point}")
    print(f"Image uri:                      {image_uri}")
    print(f"Job name:                       {job_name}")
    print(f"Hyperparameters:                {hyperparameters}")
    print(f"Instance count:                 {instance_count}")
    print(f"Instance type:                  {instance_type}")
    print(f"Queue:                          {args.queue}")
    print("#############################################################")
    print()

    estimator = PyTorch(
        entry_point=entry_point,
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        train_use_spot_instances=train_use_spot_instances,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=output_s3,
        distribution=distribution,
        max_run=max_run,
        max_wait=max_wait,
        environment=environment,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        tags=tags,
        volume_size=3000,
        enable_sagemaker_metrics=True,
        enable_remote_debug=True,
        logs=True,
    )

    if use_sm_queue:
        queue = Queue(args.queue)
        print(f"Starting VLM training job on queue: {queue.queue_name}")
        queued_jobs = queue.map(
            estimator,
            inputs=[None],
            job_names=[job_name],
            priority=args.priority,
            share_identifier=args.fss_identifier,
            timeout={"attemptDurationSeconds": max_run},
        )
        print(f"Queued jobs: {queued_jobs}")
    else:
        estimator.fit()


if __name__ == "__main__":
    main()
