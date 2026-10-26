"""Create / update the SageMaker pipeline and its daily schedule.

Run by CodeBuild (buildspec.yaml, post_build) after the Docker image is
pushed. One ProcessingStep per task in sagemaker_pipeline/config.yaml,
without dependencies between them, so they run in parallel on separate
instances and one task failing does not stop the other.
"""

import logging

import boto3
from sagemaker.network import NetworkConfig
from sagemaker.processing import Processor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_definition_config import (
    PipelineDefinitionConfig,
)
from sagemaker.workflow.retry import (
    SageMakerJobExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
)
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.triggers import PipelineSchedule

from config.config_loader import load_config
from logger.logger_setup import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

# Retry only infrastructure problems (no free GPU instance, AWS-side
# errors); a failing task (exit code 1) is not retried.
INFRA_RETRY = SageMakerJobStepRetryPolicy(
    exception_types=[
        SageMakerJobExceptionTypeEnum.CAPACITY_ERROR,
        SageMakerJobExceptionTypeEnum.INTERNAL_ERROR,
    ],
    interval_seconds=300,
    backoff_rate=2.0,
    max_attempts=3,
)


def load_pipeline_config() -> dict:
    config = load_config("sagemaker_pipeline/config.yaml")
    buildspec_config = load_config("buildspec.yaml")
    config.update(buildspec_config["env"]["variables"])

    config["docker_image_uri"] = (
        "744522205193.dkr.ecr.us-west-2.amazonaws.com/"
        + config["REPO_NAME"]
        + ":latest"
    )
    config["pipeline_name"] = config["REPO_NAME"] + "--pipeline"
    config["pipeline_schedule_name"] = config["pipeline_name"] + "-schedule"

    logger.info(f"Pipeline name: {config['pipeline_name']}")
    logger.info(f"Docker image URI: {config['docker_image_uri']}")
    return config


def build_steps(config: dict, session) -> list:
    """One ProcessingStep per task: `python3 main.py --task <task>`."""
    network_config = NetworkConfig(**config["network_config"])
    steps = []
    for task, task_cfg in config["tasks"].items():
        processor = Processor(
            role=config["role_arn"],
            image_uri=config["docker_image_uri"],
            instance_type=config["instance_type"],
            instance_count=1,
            volume_size_in_gb=config["volume_size_in_gb"],
            max_runtime_in_seconds=task_cfg["max_runtime_in_seconds"],
            base_job_name=task_cfg["step_name"],
            network_config=network_config,
            sagemaker_session=session,
        )
        steps.append(
            ProcessingStep(
                name=task_cfg["step_name"],
                processor=processor,
                job_arguments=["main.py", "--task", task],
                retry_policies=[INFRA_RETRY],
            )
        )
        logger.info(f"Step {task_cfg['step_name']}: main.py --task {task}")
    return steps


def build_pipeline(config: dict, session) -> Pipeline:
    return Pipeline(
        name=config["pipeline_name"],
        steps=build_steps(config, session),  # no depends_on: run in parallel
        sagemaker_session=session,
        pipeline_definition_config=PipelineDefinitionConfig(
            use_custom_job_prefix=True
        ),
    )


def get_existing_triggers(config: dict) -> list:
    logger.info("Checking for existing triggers...")
    try:
        client = boto3.client("scheduler")
        response = client.list_schedules(NamePrefix=config["pipeline_name"])
        triggers = [
            schedule["Name"] for schedule in response.get("Schedules", [])
        ]
        logger.info(f"Found existing triggers: {triggers}")
        return triggers
    except Exception as e:
        logger.error(f"Failed to get existing triggers: {e}")
        raise


def main():
    logger.info("Starting pipeline setup...")
    config = load_pipeline_config()

    try:
        pipeline = build_pipeline(config, PipelineSession())

        logger.info("Upserting the pipeline (creating or updating it)...")
        pipeline.upsert(role_arn=config["role_arn"])

        if get_existing_triggers(config):
            logger.info(
                f"Deleting trigger: {config['pipeline_schedule_name']}"
            )
            pipeline.delete_triggers([config["pipeline_schedule_name"]])

        if config.get("schedule"):
            logger.info(f"Scheduling pipeline with cron: {config['schedule']}")
            pipeline.put_triggers(
                triggers=[
                    PipelineSchedule(
                        name=config["pipeline_schedule_name"],
                        cron=config["schedule"],
                    )
                ],
                role_arn=config["role_arn"],
            )

        logger.info("Upserting the pipeline again with triggers...")
        pipeline.upsert(role_arn=config["role_arn"])

        if config.get("execute_on_update"):
            logger.info("Starting pipeline execution...")
            pipeline.start()
        logger.info("Pipeline setup completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline setup failed: {e}")
        raise


if __name__ == "__main__":
    main()
