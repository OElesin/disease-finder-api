import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

account_id = boto3.client('sts').get_caller_identity().get('Account')
sess = sagemaker.Session()
sagemaker_role = f'arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20181229T135829'
s3_output_location = 's3://datafy-data-lake-artifacts/disease-finder/model/'
s3_data_path = f's3://sagemaker-eu-west-1-{account_id}/data/plant-disease-dataset'

estimator = PyTorch(
    entry_point='pytorch-train-script.py',
    role=sagemaker_role,
    framework_version='1.2.0',
    train_instance_count=1,
    train_instance_type='ml.p2.xlarge',
    train_use_spot_instances=True,
    output_path=s3_output_location,
    train_max_wait=7200,
    train_volume_size=100,
    train_max_run=7200,
    sagemaker_session=sess
)

data_channels = {
    'train': f'{s3_data_path}/train',
    'valid': f'{s3_data_path}/valid'
}

estimator.fit(inputs=data_channels, wait=False)
