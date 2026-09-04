import math
from typing import List, Optional

import boto3
from botocore.config import Config
from app.config import get_settings


# R2/S3 single PutObject caps at 5 GiB; larger files must use multipart. Each
# part is min 5 MiB (except the last) and there can be at most 10000 parts. We
# use a generous part size so even very large files stay well under that count.
MIN_PART_SIZE = 256 * 1024 * 1024  # 256 MiB
MAX_PARTS = 9000


def _full_key(key: str) -> str:
    """Prepend R2_KEY_PREFIX (local-dev isolation) to a logical object key.

    Returns the key UNCHANGED when R2_KEY_PREFIX is unset -> byte-for-byte
    production behavior. When set (e.g. "dev"), every get/put/delete/multipart
    op is transparently scoped to "<prefix>/<key>", so DB-stored keys stay
    logical (unprefixed) and a delete can only ever target a prefixed key --
    it can NEVER name a bare production object. This is the single storage
    chokepoint: processing.py's raw boto3 download/upload route their keys
    through here too.
    """
    prefix = get_settings().r2_key_prefix.strip().strip("/")
    if not prefix:
        return key
    return f"{prefix}/{key}"


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
            "Key": _full_key(key),
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )


def _content_disposition(filename: str) -> str:
    """Build an attachment header, safely, from an untrusted name.

    Names reach here from user uploads and free-text edit titles and end up
    inside a quoted HTTP header, so anything that could close the quote or
    start a new header line has to go. Non-ASCII is dropped rather than
    RFC 5987-encoded: a nice filename is a courtesy on top of a working
    download, not worth risking a malformed header for.
    """
    cleaned = "".join(
        c for c in filename if c.isascii() and c.isprintable() and c not in '"\\'
    ).strip()
    return f'attachment; filename="{cleaned or "download"}"'


def generate_presigned_get(
    key: str, expires_in: int = 3600, download_as: Optional[str] = None
) -> str:
    """Signed GET for an object.

    `download_as` sets Content-Disposition on the response, which is the only
    thing that makes a browser save the file instead of rendering it. The HTML
    `download` attribute cannot do this job: it is ignored cross-origin, and R2
    is always a different origin to us, so a link to an mp4 opens the built-in
    player and the user has to go hunting through its menu to save.

    It stays opt-in because most callers here are the opposite case -- source
    playback, proxies, render previews -- where forcing a download would break
    the player outright.
    """
    settings = get_settings()
    client = _get_client()
    params = {
        "Bucket": settings.r2_bucket_name,
        "Key": _full_key(key),
    }
    if download_as:
        params["ResponseContentDisposition"] = _content_disposition(download_as)
    return client.generate_presigned_url(
        "get_object",
        Params=params,
        ExpiresIn=expires_in,
    )


def delete_object(key: str) -> None:
    settings = get_settings()
    client = _get_client()
    client.delete_object(Bucket=settings.r2_bucket_name, Key=_full_key(key))


# --- Multipart upload (for files > 5 GiB, and any large upload) ---------------

def part_size_for(file_size: int) -> int:
    """Pick a part size that keeps the part count under MAX_PARTS."""
    return max(MIN_PART_SIZE, math.ceil(file_size / MAX_PARTS))


def create_multipart_upload(key: str, content_type: str) -> str:
    settings = get_settings()
    client = _get_client()
    resp = client.create_multipart_upload(
        Bucket=settings.r2_bucket_name,
        Key=_full_key(key),
        ContentType=content_type,
    )
    return resp["UploadId"]


def generate_presigned_upload_parts(
    key: str, upload_id: str, part_count: int, expires_in: int = 86400
) -> List[str]:
    """Presign a PUT URL for each part (1..part_count). 24h expiry by default so
    slow multi-GB uploads don't outlive the signature."""
    settings = get_settings()
    client = _get_client()
    full_key = _full_key(key)
    urls: List[str] = []
    for part_number in range(1, part_count + 1):
        urls.append(
            client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": settings.r2_bucket_name,
                    "Key": full_key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=expires_in,
            )
        )
    return urls


def complete_multipart_upload(key: str, upload_id: str) -> None:
    """Finish a multipart upload. We read the part list (and ETags) server-side
    via list_parts so the browser never has to read ETag response headers --
    that sidesteps any R2 CORS expose-header configuration."""
    settings = get_settings()
    client = _get_client()
    bucket = settings.r2_bucket_name
    key = _full_key(key)

    parts: List[dict] = []
    marker = 0
    while True:
        resp = client.list_parts(
            Bucket=bucket, Key=key, UploadId=upload_id, PartNumberMarker=marker
        )
        for p in resp.get("Parts", []):
            parts.append({"PartNumber": p["PartNumber"], "ETag": p["ETag"]})
        if resp.get("IsTruncated"):
            marker = resp["NextPartNumberMarker"]
        else:
            break

    if not parts:
        raise ValueError("No uploaded parts found for this multipart upload.")

    parts.sort(key=lambda x: x["PartNumber"])
    client.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )


def abort_multipart_upload(key: str, upload_id: str) -> None:
    settings = get_settings()
    client = _get_client()
    try:
        client.abort_multipart_upload(
            Bucket=settings.r2_bucket_name, Key=_full_key(key), UploadId=upload_id
        )
    except Exception:
        pass
