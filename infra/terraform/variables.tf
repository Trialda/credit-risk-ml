variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Used as a prefix for resource naming"
  type        = string
  default     = "credit-risk-ml"
}

variable "github_repo" {
  description = "GitHub repository in 'owner/repo' form, for OIDC trust scoping"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type for the application host"
  type        = string
  default     = "t3.small"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance"
  type        = string
}