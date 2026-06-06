# One-command AWS GPU fleet (Terraform)

Provisions a single large GPU box that runs the whole worker fleet (parallel
`gpu` ingest + `cpu` render) and bootstraps it automatically. Uploads then fan
out across the GPUs.

What it creates: one EC2 GPU instance (Deep Learning AMI — NVIDIA driver +
Docker preinstalled), a default-VPC security group (egress-only unless you set
an SSH key), and an SSM role so you can shell in without a key pair. It bakes
your `.env` onto the box and `docker run`s the fleet with `--restart`.

## Prerequisites

1. **Terraform** + **AWS credentials** (`aws configure` or env vars).
2. **The worker image must include `run_workers.sh`.** CI builds the image from
   `main`, so merge `aws-migration` to `main` (or run the *Build GPU worker
   image* workflow via `workflow_dispatch`) and let it finish **before** apply.
3. A populated repo-root **`.env`** (Terraform reads it automatically).

## Run

```bash
cd deploy/aws/terraform
cp terraform.tfvars.example terraform.tfvars   # tweak if you like
terraform init
terraform apply
```

Then connect and watch it come up:

```bash
$(terraform output -raw ssm_session)   # or the ssh output if you set a key
edso-logs                              # tails the fleet
```

Healthy logs:

```
Fleet: NUM_GPUS=4 GPU_WORKERS=4 CPU_WORKERS=2 concurrency=1
ML device selected: cuda (NVIDIA A10G)
Worker ready; concurrency=1 queues=['gpu']; entering main loop.
```

Upload videos in the app — they process in parallel across the GPU workers.

## Tear down (stop paying)

```bash
terraform destroy
```

## Notes

- **Secrets live in TF state + EC2 user-data.** Fine for a demo; for production
  use SSM Parameter Store / Secrets Manager and keep state in an encrypted S3
  backend.
- **Spot** (`use_spot=true`) is cheapest but can be interrupted; the instance is
  `persistent`+`stop`, and the container restarts on resume.
- **DB connections:** each GPU worker holds a couple of Postgres connections.
  Mind Supabase's ceiling as you raise `gpu_workers` / `instance_type`.
