"""
S3 storage for reasoning_anchors outputs. An S3 URI (s3://bucket/prefix) holds the same layout as outputs/<model_tag>/:
generations/, activations/sites/, activations/trajectories/, analysis/, meta.json.

Used by extract_reasoning_states.py and reasoning_vectors.py (--s3_uri), and as a command-line tool to move a run folder
in or out of the bucket. Files whose size already matches on the other side are skipped, so an interrupted copy can
just be rerun.

    python code/s3_store.py upload   outputs/qwen3-0.6B s3://framing-interaction/reasoning_anchors/qwen3-0.6B
    python code/s3_store.py download s3://framing-interaction/reasoning_anchors/qwen3-0.6B outputs/qwen3-0.6B \
        --exclude activations/trajectories

Credentials come from the standard chain (~/.aws/credentials, AWS_PROFILE, AWS_ACCESS_KEY_ID, ...).
"""
import argparse
import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class S3Store:
    def __init__(self, uri: str):
        if not uri.startswith("s3://"):
            raise ValueError(f"not an s3:// URI: {uri}")
        self.bucket, _, prefix = uri[len("s3://"):].partition("/")
        self.prefix = prefix.strip("/")
        self.uri = f"s3://{self.bucket}/{self.prefix}"
        self.client = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))

    def key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def upload(self, local_path: str, rel: str) -> None:
        self.client.upload_file(local_path, self.bucket, self.key(rel))

    def download(self, rel: str, local_path: str) -> bool:
        """Download one object; False if it does not exist."""
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        try:
            self.client.download_file(self.bucket, self.key(rel), local_path)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):   # download_file reports a missing key as 404
                return False
            raise
        return True

    def sizes(self, rel_dir: str = "") -> dict[str, int]:
        """{path relative to the prefix: size} of every object under rel_dir."""
        base = self.key(rel_dir).rstrip("/") + "/" if (self.prefix or rel_dir) else ""
        strip = len(self.prefix) + 1 if self.prefix else 0
        out = {}
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=base):
            for obj in page.get("Contents", []):
                out[obj["Key"][strip:]] = obj["Size"]
        return out

    def upload_dir(self, local_dir: str, rel_dir: str = "", exclude: tuple[str, ...] = ()) -> int:
        remote = self.sizes(rel_dir)
        n = 0
        for root, _, files in os.walk(local_dir):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.join(rel_dir, os.path.relpath(path, local_dir)).replace(os.sep, "/").lstrip("/")
                if any(rel.startswith(e.strip("/") + "/") or rel == e.strip("/") for e in exclude):
                    continue
                if remote.get(rel) == os.path.getsize(path):
                    continue
                self.upload(path, rel)
                n += 1
        return n

    def download_dir(self, rel_dir: str, local_dir: str, exclude: tuple[str, ...] = ()) -> int:
        n = 0
        for rel, size in self.sizes(rel_dir).items():
            if any(rel.startswith(e.strip("/") + "/") or rel == e.strip("/") for e in exclude):
                continue
            path = os.path.join(local_dir, os.path.relpath(rel, rel_dir) if rel_dir else rel)
            if os.path.exists(path) and os.path.getsize(path) == size:
                continue
            self.download(rel, path)
            n += 1
        return n


def main() -> int:
    p = argparse.ArgumentParser(description="Copy a reasoning_anchors run folder to or from S3")
    p.add_argument("direction", choices=["upload", "download"])
    p.add_argument("source")
    p.add_argument("dest")
    p.add_argument("--exclude", nargs="*", default=[], help="relative paths to skip, e.g. activations/trajectories")
    args = p.parse_args()
    try:
        if args.direction == "upload":
            store = S3Store(args.dest)
            n = store.upload_dir(args.source, exclude=tuple(args.exclude))
            print(f"uploaded {n} files to {store.uri}")
        else:
            store = S3Store(args.source)
            n = store.download_dir("", args.dest, exclude=tuple(args.exclude))
            print(f"downloaded {n} files to {args.dest}")
        return 0
    except ClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
