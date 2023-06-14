from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from prefect import flow


#creating a credentials bucket block
def create_aws_cred_block():
    my_aws_cred_obj = AwsCredentials(
        aws_access_key_id = "AKIA22BLTZN7TR4HNI24",
        aws_secret_access_key = "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"
    )
    my_aws_cred_obj.save(name="ml-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("ml-aws-creds")
    my_s3_bucket_obj = S3Bucket(bucket_name="mlop-s3-bucket", credentials=aws_creds)
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)

def create_s3_block_2():
    aws_creds = AwsCredentials.load("ml-aws-creds")
    my_s3_bucket_obj = S3Bucket(bucket_name="aws-ml2-bucket", credentials=aws_creds)
    my_s3_bucket_obj.save(name="s3-bucket-2", overwrite=True)




if __name__ == "__main__":
    create_aws_cred_block()
    sleep(5)
    create_s3_bucket_block()
    sleep(2)
    create_s3_block_2()