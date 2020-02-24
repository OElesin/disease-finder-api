Disease Finder Machine Learning API | Facebook AI Hackathon - PyTorch
-----------------------------------

![Build Status](https://codebuild.eu-west-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiUDQ5ZnlQYm1QUlNNdURURlVkY0lobDR4Q0w4eitzcjNUTTRFRit5bUZjYTRkZWhieERvU1lHcHY0T1ZuVE9GWnNmcTQ3aWhadVJybGlEQndCZWNENHU0PSIsIml2UGFyYW1ldGVyU3BlYyI6ImFWYXhYeXpHd0huZkNvZkUiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)


### Description

### Built With
- **Infrastructure**: [Amazon Web Services SageMaker](https://aws.amazon.com/sagemaker/)
- **Deep Learning Library**: [PyTorch](https://pytorch.org/), transfer learning with pretrained [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)
- **Training Dataset**: [PlantVillage Disease Classification Challenge - Color Images](https://zenodo.org/record/1204914#.Xk93uBNKjPB)

### Team Members:
- [Samuel James](https://www.linkedin.com/in/samuel-james-abiodun/?originalSubdomain=de)
- [Emmanuel Adigun](https://www.linkedin.com/in/emmanuel-adigun-20202b70/?originalSubdomain=ng)
- [Olalekan Elesin](https://www.linkedin.com/in/elesinolalekan/)


### Implementation
- Model Training
    - Training Data: For this project, we made use of the [PlantVillage Disease Classification Challenge - Color Images](https://zenodo.org/record/1204914#.Xk93uBNKjPB). Containing
    38 classes of plant diseases. One the limitations of the dataset is that it only contains images of leaves, as such is not valuable when predicting diseases on fruits or farm produce.
    This limitation for us informs the next step of DiseaseFinder. 
    - Deep Learning Model: Our starting point was to follow the [PyTorch image classification tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). We made use of
    ResNet18 from the PyTorch model zoo.
    - Model Training Infrastructure: Our model was trained on [Amazon SageMaker Notebooks](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) [GPU instance, ml.p2.xlarge](https://aws.amazon.com/sagemaker/pricing/instance-types/).
    Once the model was trained on GPU, we saved the model as a pickle object in both [GPU](./api/model_dir/plant-disease-model-gpu.pt) and [CPU](./api/model_dir/plant-disease-model-cpu.pt) versions.

- Model Deployment: For the purposes of the hackathon, we realized that it would be expensive to have real-time inference hosted on AWS SageMaker. Hence,
we decided to host the model with [Heroku Containers](https://devcenter.heroku.com/articles/container-registry-and-runtime), however, this poses latency challenges. Once we see traction with the solution,
deployment on Amazon SageMaker would be next. In addition to hosting the model on Heroku, we implemented a continuous deployment pipeline with [Amazon CodeBuild](https://aws.amazon.com/codebuild/), with the goal of embracing [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html).
    

### Challenges:
- Currently, we had access to dataset containing images of leaves. This alone limits the performance of the model as it would not be able to classify the fruits. We hope to collect data on the fruits and improve on the performance of the model.
- We would also want to move from only classifying diseases to actually detecting spots with [object detection](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) or further improvement with [instance segmentation](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- Expensive but affordable to train on GPU due to infrastructure cost. This is because GPU even though on the cloud do not come cheap.
- Deployment might also be expensive. However, we hosted the model for free on [Heroku](https://www.heroku.com/). If the project at some point starts generating revenue, we will move to scale the deployment on [AWS SageMaker](https://aws.amazon.com/sagemaker/).


### Model Usage
Below is an example API request to the deployed deep learning model
```bash
$ curl -XPOST -H "Content-Type: application/json" -d '{"ImageUrl": "https://www.sciencesource.com/Doc/TR1_WATERMARKED/7/5/4/f/SS2839121.jpg?d63644905319" }' https://disease-finder-api.herokuapp.com/invocations

{"prediction": "Cherry (including sour) healthy"} 
```