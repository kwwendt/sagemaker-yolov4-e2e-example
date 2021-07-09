# Amazon SageMaker - YOLOv4 End-to-End Example Notebook

## Clone this repository
```
git clone https://github.com/kwwendt/sagemaker-yolov4-e2e-example.git
```
## Training
For this example, you will need to supply your own dataset for training. There are free CV datasets available on Roboflow and other similar sites.

### Building the custom training container
Make sure you have AWS CLI access to interact with Amazon Elastic Container Registry.

Execute the following commands:
```
cd training
```
```
chmod +x build_and_push_container.sh
./build_and_push_container.sh
```

This may take several minutes. Please wait until the container is fully pushed up to ECR to continue.

### Execute the notebook
Execute the steps within the `e2e_yolov4_training.ipynb` notebook in Amazon SageMaker Studio.