output "instance_ids" {
  description = "EC2 instance ids (one per GPU box)."
  value       = aws_instance.fleet[*].id
}

output "public_ips" {
  description = "Public IPs of the worker boxes."
  value       = aws_instance.fleet[*].public_ip
}

output "ssm_sessions" {
  description = "Shell into each box without SSH (needs AWS CLI + Session Manager plugin)."
  value       = [for id in aws_instance.fleet[*].id : "AWS_PROFILE=${var.aws_profile} aws ssm start-session --region ${var.region} --target ${id}"]
}

output "ssh" {
  description = "SSH commands (only if ssh_key_name was set)."
  value = var.ssh_key_name != "" ? [
    for ip in aws_instance.fleet[*].public_ip : "ssh ubuntu@${ip}"
  ] : ["(no key pair set; use ssm_sessions)"]
}

output "tail_logs" {
  description = "After connecting to a box, watch the fleet boot."
  value       = "edso-logs   # or: docker logs -f edso-fleet"
}
