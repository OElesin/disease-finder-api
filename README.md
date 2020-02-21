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


### Challenges:
- Currently, we had access to dataset containing images of leaves. This alone limits the performance of the model as it would not be able to classify the fruits. We hope to collect data on the fruits and improve on the performance of the model.
- Expensive but affordable to train on GPU due to infrastructure cost. This is because GPU even though on the cloud do not come cheap.
- Deployment might also be expensive. However, we hosted the model for free on [Heroku](https://www.heroku.com/). If the project at some point starts generating revenue, we will move to scale the deployment on [AWS SageMaker](https://aws.amazon.com/sagemaker/).


### Model Usage
```bash
$ curl -XPOST -H "Content-Type: application/json" -d '{ "ImageUrl": "https://www.sciencesource.com/Doc/TR1_WATERMARKED/7/5/4/f/SS2839121.jpg?d63644905319" }' https://disease-finder-api.herokuapp.com/invocations 
```