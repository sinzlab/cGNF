# Conditional Graph Normalizing Flows
This is the official implementation of the paper "Conditional Graph Normalizing Flows"

<p align="center"><img src="figure/pipeline.png" width="100%" alt="Pipeline for Human Pose Estimation with Conditional Graph Normalizing Flows (cGNFs)" /></p>

> [**Conditional Graph Normalizing Flows**](),\
> Pierzchlewicz, P. A., Cotton, R. J., Bashiri, M. & Sinz, F. H.

## Installation
### Prerequisites
This project requires that you have the following installed:
- [docker](https://docs.docker.com/get-docker/)
- [docker-compose](https://docs.docker.com/compose/install/)

Ensure that you have the base image pulled from the Docker Hub.
You can get the base image by running the following command:
```shell
docker pull sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
```

### Step-by-step Guide
1. Clone the repository.
2. Navigate to the project directory. 
3. Build the environment with `docker-compose build base`.

## Model Evaluation
You can evaluate the model with the following command:
```
docker-compose run eval --human36m --experiment=mpii-xlarge
```

### Available Pretrained Models
| Model Name                  | description                                                                    | Artifact path                                | Weights |
|-----------------------------|--------------------------------------------------------------------------------|----------------------------------------------| ------- |
| Extra Large cGNF Human 3.6m | Extra large model trained on the Human 3.6M dataset with MPII input keypoints. | ```ppierzc/cgnf/cgnf_human36m-xlarge:best``` | [link](https://wandb.ai/ppierzc/propose_human36m/artifacts/model/mpii-prod-xlarge/v20/files) |
| Large cGNF Human 3.6m       | Large model trained on the Human 3.6M dataset with MPII input keypoints.       | ```ppierzc/cgnf/cgnf_human36m-large:best```  | [link](https://wandb.ai/ppierzc/propose_human36m/artifacts/model/mpii-prod-large/v20/files) |
| cGNF Human 3.6m             | Model trained on the Human 3.6M dataset with MPII input keypoints.             | ```ppierzc/cgnf/cgnf_human36m:best```        | [link](https://wandb.ai/ppierzc/propose_human36m/artifacts/model/mpii-prod/v20/files) |
