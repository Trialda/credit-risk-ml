output "ec2_public_ip" {
  description = "Public IP of the application instance"
  value       = aws_instance.app.public_ip
}

output "ecr_backend_url" {
  description = "ECR repository URL for the backend image"
  value       = aws_ecr_repository.backend.repository_url
}

output "ecr_frontend_url" {
  description = "ECR repository URL for the frontend image"
  value       = aws_ecr_repository.frontend.repository_url
}

output "github_actions_role_arn" {
  description = "IAM role ARN GitHub Actions assumes via OIDC"
  value       = aws_iam_role.github_actions.arn
}