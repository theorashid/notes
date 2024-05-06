---
tags:
  - swe
folder: learning
share: true
title: infrastructure as code
date created: Friday, May 3rd 2024, 4:47:56 pm
date modified: Friday, May 3rd 2024, 5:22:28 pm
---

Managing and provisioning computer resources through files (rather than configure in AWS).

## terraform

- **VPC** (virtual private cloud), isolated network

```tf  
resource "aws_vpc" "main" {  
  cidr_block       = "[10.0.0.0/16](http://10.0.0.0/16)"  
  instance_tenancy = "default"  
  
  tags = {  
    Name = "test-db"  
  }  
}  
```  

- Public **subnet**
	- range of IP addresses in your VPC. A subnet must reside in a single Availability Zone.

```tf  
resource "aws_subnet" "main" {  
  vpc_id     = [aws_vpc.main.id](http://aws_vpc.main.id/)  
  cidr_block = "[10.0.1.0/24](http://10.0.1.0/24)"  
  
  tags = {  
    Name = "test-db"  
  }  
}  
```  

- Security group rule(s)
	- ingress/egress from IP of server & platform IPs
	- control what's coming in an out  

```tf  
resource "aws_security_group_rule" "main" {  
  type              = "ingress"  
  from_port         = 5432  
  to_port           = 5432  
  protocol          = "tcp"  
  cidr_blocks       = [aws_vpc.main.cidr_block]  
  ipv6_cidr_blocks  = [aws_vpc.main.ipv6_cidr_block]  
  security_group_id = "sg-123456"  
}  
```  

- Instance e.g. RDS

```tf  
resource "aws_db_instance" "default" {  
  allocated_storage    = 10  
  db_name              = "testdb"  
  engine               = "postgres"  
  engine_version       = "12"  
  instance_class       = "db.t3.micro"  
  username             = "foo"  
  password             = "foobarbaz"  
  skip_final_snapshot  = true  
}  
```

Commands:

- `terraform plan`. View what needs to be done
- `terraform apply`. Apply any changes
- `terraform destroy`. Destroy any resources once you are done

To deploy in CI/CD:

```yaml  
variables:  
  TF_STATE_NAME: default  
  TF_CACHE_KEY: default  
  TF_ROOT: ${CI_PROJECT_DIR}/deployment  
  TF_ADDRESS: ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/terraform/state/${CI_COMMIT_BRANCH}  
  AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID  
  AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY  
  
stages:  
- plan  
- deploy  
- destroy  
  
terraform:plan:  
  stage: plan  
  image: registry.gitlab.com/gitlab-org/terraform-images/stable:latest 
  script:  
    - cd "${TF_ROOT}"  
    - gitlab-terraform plan  
    - gitlab-terraform plan-json  
  cache:  
    policy: pull  
  artifacts:  
    name: plan  
    paths:  
      - ${TF_ROOT}/plan.cache  
  
terraform:deploy:  
  stage: deploy  
  image: registry.gitlab.com/gitlab-org/terraform-images/stable:latest 
  script:  
    - cd "${TF_ROOT}"  
    - gitlab-terraform apply  
  artifacts:  
    paths:  
      - ${TF_ROOT}/plan.cache  
  rules:  
    - when: manual  
  needs:  
    - job: terraform:plan  
  
terraform:destroy:  
  stage: destroy  
  image: registry.gitlab.com/gitlab-org/terraform-images/stable:latest 
  script:  
    - cd "${TF_ROOT}"  
    - gitlab-terraform destroy  
  artifacts:  
    paths:  
      - ${TF_ROOT}/plan.cache  
  rules:  
    - when: manual  
  needs:  
    - job: terraform:deploy  
```
