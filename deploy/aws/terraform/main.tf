locals {
  # Default to the repo-root .env (deploy/aws/terraform -> ../../../.env).
  env_file = var.env_file_path != "" ? var.env_file_path : "${path.module}/../../../.env"
  env_b64  = fileexists(local.env_file) ? base64encode(file(local.env_file)) : ""
}

# GPU-ready AMI: NVIDIA driver + nvidia-container-toolkit + Docker preinstalled.
data "aws_ssm_parameter" "gpu_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

# Use the account's default VPC/subnets so there's nothing else to provision.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- networking: egress-only by default; SSH only if a key is provided -------
resource "aws_security_group" "fleet" {
  name_prefix = "${var.name}-"
  description = "EDSO worker fleet: outbound only (jobs pulled from Postgres/R2)."
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = var.ssh_key_name != "" ? [1] : []
    content {
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [var.allowed_ssh_cidr]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = var.name }
}

# --- IAM: SSM Session Manager so you can shell in without a key pair ---------
resource "aws_iam_role" "fleet" {
  name_prefix = "${var.name}-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.fleet.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "fleet" {
  name_prefix = "${var.name}-"
  role        = aws_iam_role.fleet.name
}

# --- the box -----------------------------------------------------------------
resource "aws_instance" "fleet" {
  ami                         = data.aws_ssm_parameter.gpu_ami.value
  instance_type               = var.instance_type
  subnet_id                   = element(data.aws_subnets.default.ids, 0)
  vpc_security_group_ids      = [aws_security_group.fleet.id]
  iam_instance_profile        = aws_iam_instance_profile.fleet.name
  associate_public_ip_address = true
  key_name                    = var.ssh_key_name != "" ? var.ssh_key_name : null

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    image              = var.image
    env_b64            = local.env_b64
    gpu_workers        = var.gpu_workers
    cpu_workers        = var.cpu_workers
    worker_concurrency = var.worker_concurrency
    ghcr_user          = var.ghcr_user
    ghcr_pat           = var.ghcr_pat
  })

  tags = { Name = var.name }
}
