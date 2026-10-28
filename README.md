# Chaperone-SA-1.0 on Amazon SageMaker AI

Chaperone-SA-1.0 reads pharma and medical text and says, for each drug or therapy you name, whether the text presents
it as Positive, Neutral or Negative. On AWS Marketplace it is an Amazon SageMaker AI model package that you deploy in
your own AWS account, as a real-time endpoint or as batch transform jobs. Your texts stay in your account: SageMaker AI
runs the model without network access, and nothing is stored.

This repository holds the sample notebook, a Python client and the sample data of the AWS Marketplace listing.

| File | What |
|---|---|
| [`chaperone-on-sagemaker.ipynb`](chaperone-on-sagemaker.ipynb) | Deploys Chaperone-SA-1.0 on ml.g6e.2xlarge, analyzes texts in real time, runs a batch transform job, and deletes what it made |
| [`chaperone_sagemaker.py`](chaperone_sagemaker.py) | A Python client: builds each request, splits long target lists, parses and checks the answers, retries an unreadable one |
| [`samples/request.json`](samples/request.json) → [`samples/response.json`](samples/response.json) | A real-time request and its answer |
| [`samples/input.jsonl`](samples/input.jsonl) → [`samples/output.jsonl`](samples/output.jsonl) | A batch transform job's input and output |

The texts and drug names in the samples are fictional. The answers in `samples/` are the model's own, from
ml.g6e.2xlarge.

## Before you start

1. Subscribe to Chaperone-SA-1.0 on AWS Marketplace.
2. Check your SageMaker AI quotas in Service Quotas: *ml.g6e.2xlarge for endpoint usage* for real-time endpoints,
   *ml.g6e.2xlarge for transform job usage* for batch transform. New accounts often have 0; ask for at least 1.
3. Open the notebook in SageMaker AI Studio or on a notebook instance whose role may create SageMaker AI models,
   endpoints and transform jobs, and use SageMaker AI's default S3 bucket (for example with the managed policy
   AmazonSageMakerFullAccess).

## Instance types

The model runs on one NVIDIA GPU with at least 40 GB of memory and sizes itself to it at startup:

| Instance types | GPU | Tokens per request (text and answer) | Requests at once |
|---|---|---:|---:|
| ml.g6e.2xlarge (recommended), .4xlarge, .8xlarge, .16xlarge | 1 NVIDIA L40S, 48 GB | 8,192 | 16 |
| ml.g7e.2xlarge, .4xlarge, .8xlarge (real-time only) | 1 NVIDIA RTX PRO 6000 Blackwell, 96 GB | 32,768 | 64 |
| ml.p5.4xlarge (real-time only) | 1 NVIDIA H100, 80 GB | 32,768 | 64 |

Copying the 28 GB of weights and starting the model take longer than SageMaker AI's defaults allow, so give an
endpoint's production variant 30 minutes for each: `ModelDataDownloadTimeoutInSeconds` and
`ContainerStartupHealthCheckTimeoutInSeconds` of 1800. A new endpoint takes about 40 minutes to start.

## Requests

The endpoint takes an OpenAI chat-completions body (`application/json`). Build it the way the model was trained, or
its answers get worse. `chaperone_sagemaker.py` and the notebook do this for you:

| Field | Value |
|---|---|
| `messages[0]` | `system`: the system prompt `SYSTEM_PROMPT` in `chaperone_sagemaker.py`, word for word. Your own instructions (up to 2,000 characters) may follow it, after a blank line and `Additional instructions:`, and then `INSTRUCTIONS_GUARD`. |
| `messages[1]` | `user`: `Source type: <type>`, a blank line, `Content:`, the text between lines `<<<` and `>>>`, a blank line, `Target topics:` and the numbered targets, each with its aliases as `(also: alias1; alias2)` |
| `temperature`, `top_p` | 0 and 1 |
| `max_tokens` | 32 plus, for each target, 16 + half its label's length; at most 512 |
| `chat_template_kwargs` | `{"enable_thinking": false}` |
| `response_format` | A JSON schema with one required `Positive`/`Neutral`/`Negative` property per target, as in [`samples/request.json`](samples/request.json) |

The source type is one of `Conference abstract`, `Publication (PubMed record)`, `Clinical trial registry record
(ClinicalTrials.gov)`, `Social media post by a physician`, `Press / news article`, `Case report sentence (MEDLINE)` or
`Other`. Give each request 1 to 8 targets, and split longer lists into several requests.

The answer is an OpenAI chat completion. `choices[0].message.content` holds a JSON object with exactly the requested
targets as keys, for example `{"Zentrivo": "Positive", "Oravexa": "Negative"}`, and `usage` counts the tokens.

```python
from chaperone_sagemaker import Analyzer, SageMakerTransport, Topic

analyzer = Analyzer(SageMakerTransport("my-chaperone-endpoint"))
result = analyzer.analyze(
    "In the phase 3 VERANTA trial, zentrimab prolonged progression-free survival compared with Oravexa (HR 0.61).",
    [Topic("Zentrivo", ["zentrimab"]), Topic("Oravexa")],
    source_type="Conference abstract",
    match="mentioned",   # skip targets the text doesn't name; "all" (the default) asks about every target
)
print(result.topics, result.error)
```

## With the AWS CLI

`MODEL_PACKAGE_ARN` is the model package of your subscription in your Region, and `ROLE_ARN` a role SageMaker AI can
assume:

```bash
aws sagemaker create-model --model-name chaperone --execution-role-arn "$ROLE_ARN" \
  --primary-container ModelPackageName="$MODEL_PACKAGE_ARN" --enable-network-isolation

# a real-time endpoint
aws sagemaker create-endpoint-config --endpoint-config-name chaperone --production-variants \
  VariantName=all,ModelName=chaperone,InstanceType=ml.g6e.2xlarge,InitialInstanceCount=1,ModelDataDownloadTimeoutInSeconds=1800,ContainerStartupHealthCheckTimeoutInSeconds=1800
aws sagemaker create-endpoint --endpoint-name chaperone --endpoint-config-name chaperone
aws sagemaker wait endpoint-in-service --endpoint-name chaperone
aws sagemaker-runtime invoke-endpoint --endpoint-name chaperone --content-type application/json \
  --body fileb://samples/request.json response.json

# a batch transform job: one request per line; every file under input/ gets its answers in output/<file>.out
aws sagemaker create-transform-job --transform-job-name chaperone-batch --model-name chaperone \
  --batch-strategy SingleRecord --max-payload-in-mb 6 --max-concurrent-transforms 1 \
  --transform-input 'DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/input/}},ContentType=application/json,SplitType=Line' \
  --transform-output 'S3OutputPath=s3://<bucket>/output/,Accept=application/json,AssembleWith=Line' \
  --transform-resources InstanceType=ml.g6e.2xlarge,InstanceCount=1
```

Batch transform needs `SingleRecord` and `SplitType` `Line`: the model takes one request per call. For large files,
raise `--max-concurrent-transforms` up to 16, the number of requests the model serves at once on ml.g6e.

## Limits

- English text. Training covered documents of up to about 4,000 tokens; on ml.g6e a request with its answer may hold
  8,192 tokens, so split longer documents into passages.
- One to 8 targets per request.
- The model judges only what the text says about each target. Its results can be wrong: review them before you rely on
  them.

For research and commercial-intelligence use. Chaperone-SA-1.0 isn't a medical device and isn't for diagnosis or
treatment decisions, and it doesn't replace pharmacovigilance.

## License

The code and sample data in this repository are under the [MIT License](LICENSE). Chaperone-SA-1.0 itself is
proprietary to Empirisch Tech GmbH and licensed through your AWS Marketplace subscription.

Support: contact@empirischtech.at, [chaperoneai.net/contact](https://chaperoneai.net/contact)
