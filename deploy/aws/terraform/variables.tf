variable "region" {
  description = "AWS region. Must match where your G-instance quota lives (us-east-1 here)."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile to use. Empty = default credential chain."
  type        = string
  default     = "edso"
}

variable "name" {
  description = "Name tag / prefix for created resources."
  type        = string
  default     = "edso-worker-fleet"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance (per box). Under the 16-vCPU Spot G quota:
      g5.2xlarge  = 8 vCPU, 1x A10G 24GB  -> 2 boxes = 16 vCPU = 2 GPUs (chosen)
      g5.4xlarge  = 16 vCPU, 1x A10G      -> 1 box only
      g4dn.xlarge = 4 vCPU, 1x T4 16GB    -> 4 boxes (slower, OOM risk)
  EOT
  type        = string
  default     = "g5.2xlarge"
}

variable "worker_count" {
  description = "Number of GPU boxes. 2x g5.2xlarge = 16 vCPU = the Spot G quota ceiling."
  type        = number
  default     = 2
}

variable "use_spot" {
  description = "Use spot. On-Demand G quota is 0 in us-east-1, so this must stay true."
  type        = bool
  default     = true
}

variable "root_volume_gb" {
  description = "Root EBS size (gp3). Holds the ~7GB of model weights + transient proxies."
  type        = number
  default     = 200
}

variable "image" {
  description = "Worker container image (built by .github/workflows/build-worker.yml)."
  type        = string
  default     = "ghcr.io/vivek2643/cloud-worker:latest"
}

variable "gpu_workers" {
  description = "GPU-queue worker processes. Empty = auto (1 per detected GPU)."
  type        = string
  default     = ""
}

variable "cpu_workers" {
  description = "CPU-queue (render) worker processes per box. 1 is plenty on an 8-vCPU box."
  type        = number
  default     = 1
}

variable "worker_concurrency" {
  description = "Per-process concurrency. Keep 1 to avoid VRAM contention."
  type        = number
  default     = 1
}

variable "env_file_path" {
  description = "Path to the .env whose secrets are baked into the box. Defaults to the repo-root .env."
  type        = string
  default     = "" # resolved to <repo>/.env in main.tf when empty
}

variable "ghcr_user" {
  description = "GitHub username for GHCR login. Leave empty if the image package is public."
  type        = string
  default     = ""
}

variable "ghcr_pat" {
  description = "GHCR read:packages PAT. Leave empty if the image package is public."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ssh_key_name" {
  description = "Existing EC2 key pair name for SSH. Leave empty to use SSM Session Manager only (no inbound port)."
  type        = string
  default     = ""
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH (only used when ssh_key_name is set). Lock this to your IP."
  type        = string
  default     = "0.0.0.0/0"
}
