#!/bin/bash

# Set your AWS account ID and region
AWS_ACCOUNT_ID="your-aws-account-id"
AWS_REGION="your-aws-region"

# Set your ECR repository name
ECR_REPO_NAME="your-ecr-repo-name"

# Build the Docker image
docker build -t $ECR_REPO_NAME .

# Authenticate Docker to your ECR registry
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag the image
docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

# Push the image to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest