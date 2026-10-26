# ds-ask-sa-chat-analysis

Daily analysis of Ask SA chats with an LLM (`google/gemma-2-9b-it` on one L4 GPU).
Two independent tasks, each its own SageMaker pipeline step:

| Task | What it does | Target table |
|---|---|---|
| `sentiment` | Classifies every user follow-up message (label, subtype, confidence) | `ds.production__ask_sa_sentiment_followup_msg_analysis` |
| `ner` | Extracts tickers, sectors, industries and topics from the first message of each chat: dictionary/fuzzy matching finds candidates, the LLM keeps the intended ones | `ds.production__ask_sa_message_entity_extraction` |

## Run

```bash
python main.py --task sentiment
python main.py --task ner
python main.py --task ner --dry-run --max-rows 50   # everything except writing to S3 / Redshift
```

Exit code `0` = success (also when there is nothing new), `1` = failed.
Run one task per process: the model needs ~18 GB of the 22 GB GPU.

## Layout

```
main.py                     entry point (--task)
variables.yaml              S3, Redshift, secret names, table names
queries/                    get_chats.sql (sentiment), get_all_user_chats.sql + get_tickers.sql (ner)
src/asksa_jobs/             job orchestration: common.py (config, secrets, Redshift, S3 + COPY),
                            sentiment_job.py, ner_job.py
src/asksa_follow_up/        follow-up extraction + LLM classifier (FollowupLLM)
src/asksa_entity/           candidate matching (baseline.py, ticker_noise.py) + LLM validation
                            (llm_validate.py, EntityValidatorLLM)
notebooks/                  sentiment_tests.ipynb, ner_tests.ipynb: verification on a GPU instance
sagemaker_pipeline/         pipeline definition and schedule (deployed by CodeBuild)
```

Settings for each job are in `SentimentParams` / `NerParams`; their docstrings explain
why values differ from the original notebooks.

## Secrets

Redshift credentials and the Hugging Face token are read from AWS Secrets Manager
(`REDSHIFT_SECRET_NAME`, `HF_SECRET_NAME` in `variables.yaml`). Never put tokens in this repo.

## Deploy

Push to the repo: CodeBuild (`buildspec.yaml`) builds the Docker image, pushes it to ECR
and upserts the SageMaker pipeline (`sagemaker_pipeline/sm_pipeline.py`).
