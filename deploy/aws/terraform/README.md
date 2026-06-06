# AWS GPU spot fleet (Terraform)

Provisions a fleet of GPU boxes that each run a worker (parallel `gpu` ingest +
`cpu` render) and bootstrap themselves. Uploads fan out across the boxes.

Defaults match the approved setup: **us-east-1, Spot-only, 2x g5.2xlarge =
2x A10G = 2 videos in parallel** (On-Demand G quota is 0 here; Spot G is 16
vCPU, and 2x g5.2xlarge = 16 vCPU is the ceiling).

What it creates per box: an EC2 GPU instance (Deep Learning AMI — NVIDIA driver
+ Docker preinstalled), a shared egress-only security group, and an SSM role so
you can shell in without a key pair. It bakes your `.env` onto the box and
`docker run`s the fleet with `--restart`.

## Prerequisites

1. **Terraform** + **AWS credentials** (profile `edso` by default).
2. **The worker image must include `run_workers.sh`.** Trigger the *Build GPU
   worker image* GitHub Action on the `aws-migration` branch (Actions -> Run
   workflow) and let it finish **before** apply.
3. **Make the GHCR package public** (Packages -> `cloud-worker` -> Package
   settings -> Change visibility -> Public) so the boxes can pull it. (Or set
   `ghcr_user`/`ghcr_pat` to keep it private.)
4. A populated repo-root **`.env`** (Terraform reads it automatically).

## Run

```bash
cd deploy/aws/terraform
cp terraform.tfvars.example terraform.tfvars   # tweak if you like
AWS_PROFILE=edso terraform init
AWS_PROFILE=edso terraform apply
```

Then connect to a box and watch it come up:

```bash
terraform output ssm_sessions   # prints a start-session command per box
# run one of them, then:
edso-logs                       # tails the fleet (docker logs -f edso-fleet)
```

Healthy logs (per box, 1 GPU each):

```
Fleet: NUM_GPUS=1 GPU_WORKERS=1 CPU_WORKERS=1 concurrency=1
ML device selected: cuda (NVIDIA A10G)
Worker ready; concurrency=1 queues=['gpu']; entering main loop.
```

Upload videos in the app — they process two-at-a-time across the boxes.

## Tear down (stop paying)

```bash
AWS_PROFILE=edso terraform destroy
```

## Notes

- **Spot-only here.** On-Demand G quota is 0 in us-east-1, so `use_spot=true` is
  required. Requests are `persistent`+`stop`, and the container restarts on
  resume — but spot can still be reclaimed. For rock-solid uptime, request an
  On-Demand G quota increase later.
- **Secrets live in TF state + EC2 user-data.** Fine for a demo; for production
  use SSM Parameter Store / Secrets Manager and an encrypted S3 state backend.
- **DB connections:** each GPU worker holds a couple of Postgres connections.
  Mind Supabase's ceiling as you raise `worker_count`.
