import boto3
from botocore.config import Config
from app.config import get_settings


def _get_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def generate_presigned_put(key: str, content_type: str, expires_in: int = 3600) -> str:
    settings = get_settings()
    client = _get_client()
    return client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.r2_bucket_name,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )


def generate_presigned_get(key: str, expires_in: int = 3600) -> str:
    settings = get_settings()
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.r2_bucket_name,
            "Key": key,
        },
        ExpiresIn=expires_in,
    )


def delete_object(key: str) -> None:
    settings = get_settings()
    client = _get_client()
    client.delete_object(Bucket=settings.r2_bucket_name, Key=key)
