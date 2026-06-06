output "instance_id" {
  description = "EC2 instance id."
  value       = aws_instance.fleet.id
}

output "public_ip" {
  description = "Public IP of the worker box."
  value       = aws_instance.fleet.public_ip
}

output "ssm_session" {
  description = "Shell in without SSH (needs AWS CLI + Session Manager plugin)."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.fleet.id}"
}

output "ssh" {
  description = "SSH command (only if ssh_key_name was set)."
  value       = var.ssh_key_name != "" ? "ssh ubuntu@${aws_instance.fleet.public_ip}" : "(no key pair set; use ssm_session)"
}

output "tail_logs" {
  description = "After connecting, watch the fleet boot."
  value       = "edso-logs   # or: docker logs -f edso-fleet"
}
