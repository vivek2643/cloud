variable "region" {
  description = "AWS region. Keep close to Supabase (their DB is us-west-2) to cut latency."
  type        = string
  default     = "us-west-2"
}

variable "name" {
  description = "Name tag / prefix for created resources."
  type        = string
  default     = "edso-worker-fleet"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance. One large box:
      g5.2xlarge   = 1x A10G 24GB  (cheapest, first parallel test)
      g5.12xlarge  = 4x A10G 24GB  (GPU_WORKERS=4, 4 videos at once)
      g6.12xlarge  = 4x L4  24GB   (newer/cheaper alt)
  EOT
  type        = string
  default     = "g5.12xlarge"
}

variable "use_spot" {
  description = "Use a spot instance (~60-70% cheaper, can be interrupted). On-demand is safer mid-demo."
  type        = bool
  default     = false
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
  description = "CPU-queue (render) worker processes."
  type        = number
  default     = 2
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
