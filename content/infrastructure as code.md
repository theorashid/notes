---
tags:
  - swe
folder: learning
share: true
title: infrastructure as code
date created: Friday, May 3rd 2024, 4:47:56 pm
date modified: Monday, January 13th 2025, 10:07:32 pm
---

Managing and provisioning computer resources through files (rather than configure in AWS).

Compare AWS CDK (TypeScript) and [Terraform](https://www.terraform.io/) (HashiCorp configuration language).

AWS CDK elements should be within `App`.

```ts
import { App, Stack } from 'aws-cdk-lib';
import { Construct } from 'constructs';

interface CustomProps {
  readonly stuff: ...;
}

export class NewStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: CustomProps) {
    super(scope, id, {
      // stuff from CustomProps
    });

    // define stack elements
  }
}

const app = new App();
```

## VPC (virtual private cloud)

Isolated network with subnets.

```ts
const vpc = new ec2.Vpc(this, 'VPCName', {
  maxAzs: 2,
  restrictDefaultSecurityGroup: true
  subnetConfiguration: [
    {
	  cidrMask: 24,
	  name: 'public',
	  subnetType: ec2.SubnetType.PUBLIC,
	},
	{
	  cidrMask: 24,
	  name: 'private',
	  subnetType: ec2.SubnetType.PRIVATE_WITH_NAT,
	},
  ],
})
```

```hcl
resource "aws_vpc" "vpc_name" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "MyVPC"
  }
}

# define subnets separately to VPC
resource "aws_subnet" "public" {
  count                   = 3
  vpc_id                  = aws_vpc.vpc_name.id
  cidr_block              = "10.0.0.0/24"

  tags = {
    Name = "public-vpc"
  }
}

resource "aws_subnet" "private" {
  count                   = 3
  vpc_id                  = aws_vpc.vpc_name.id
  cidr_block              = "10.0.0/24"

  tags = {
    Name = "private-vpc"
  }
}
```  

## security group  

Control IP traffic coming in (ingress) and out (egress).

```ts
const customSecurityGroup = new ec2.SecurityGroup(this, 'SecurityGroupName', {
  vpc,
  description: 'what the security group is for',
  allowAllOutbound: Exempt(true),
});
securityGroup.addIngressRule(
  ec2.Peer.anyIpv4(),
  ec2.Port.tcp(22),  // Port 442 for HTTPS, Port 5432 for database
  'Allow SSH access from anywhere',
);
securityGroup.addIngressRule(
  securityGroup,
  ec2.Port.tcp(5432),
  'Allow PostgreSQL metadata database acesss',
)
```

```hcl  
resource "aws_security_group" "custom_security_group" {
  vpc_id = aws_vpc.my_vpc.id
  description = "what the security group is for"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow SSH access from anywhere"
  }

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.custom_security_group.id]
    description     = "Allow PostgreSQL metadata database access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "CustomSecurityGroup"
  }
}
```  

## iam roles

```ts
const sagemakerRole = new iam.Role(this, 'sagemakerRole', {
  assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
  managedPolicies: [
    iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess')
  ]
})
```

```hcl
resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker_role"
  path = "/"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_role.json
  managed_policy_arns = ["arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"]
}

data "aws_iam_policy_document" "sagemaker_role" {
  statement {
  actions = ["sts:AssumeRole"]
  principals {
    type = "Service"
    identifiers = ["sagemaker.amazonaws.com"]
  }
}
```

## bucket

```typescript
const bucket = new s3.Bucket(this, 'BucketName' {
  versioned: true,
  encryption: s3.BucketEncryption.S3_MANAGED,
  removalPolicy: RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
  enforceSSL: true,
})
```

```hcl
resource "aws_s3_bucket" "my_bucket" {
  bucket = "bucket-name"
  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  lifecycle {
    prevent_destroy = false
  }

  lifecycle_rule {
    id = "autoDeleteObjects"
    enabled = true
    noncurrent_version_expiration {
      days = 30
    }
  }

  force_destroy = true
  enforce_ssl = true
}
```

## database

```ts
const database = new rds.DatabaseInstance(this, 'MyDatabase', {
  engine: rds.DatabaseInstanceEngine.postgres,
  vpc: vpc,
  securityGroup: securityGroup,
  publiclyAccessible: true,
  allocatedStorage: 20,
  maxAllocatedStorage: 100,
  instanceClass: 'db.t3.micro',
  databaseName: 'testdb',
  masterUser: 'admin',
  masterUserPassword: cdk.SecretValue.secretsManagerSecret(this, 'MyDatabaseSecret'),
});
```

```hcl
resource "aws_db_instance" "my_database" {
  db_name                = "testdb"
  engine                 = "postgres"
  vpc_security_group_ids = [aws_security_group.custom_security_group.id]
  publicly_accessible.   = true
  allocated_storage      = 20
  max_allocated_storage  = 100
  instance_class         = "db.t3.micro"
  database_name          = "my_database"
  username               = "foo"
  password               = "foobarbaz"

  tags = {
    Name = "testdb"
  }
}
```

## mlflow tracking server

```ts
const mlflowTrackingServer = new sagemaker.CfnMlflowTrackingServer(this, 'TrackingServer', {
  artifactStoreUri: `s3://${bucket.bucketName}`,
  roleArn: sagemakerRole.roleArn,
  trackingServerName: 'TrackingServer',
  automaticModelRegistration: true,
  mlflowVersion: '2.13.2',
  trackingServerSize: 'Medium',
})
```

```hcl
resource "aws_instance" "mlflow_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.medium"

  tags = {
    Name = "MLflow Tracking Server"
  }
}
```

## kubernetes (EKS) cluster

```ts
const cluster = new eks.Cluster(this, 'EKSCluster', {
  // vpc: vpc, // created by default
  // securityGroup: securityGroup, // for control plane, created by default
  defaultCapacity: 0, // start with 0 capacity and add
  version: eks.KubernetesVersion.V1_31,
  // secretsEncrpytionKey: // kms key
  // kubectlLayer: KubectlV31Layer(),
  ipFamily: eks.IpFamily.IP_V4,
  enpointAccess: eks.EndpointAccess.PUBLIC_AND_PRIVATE,
  clusterLogging: [
    ClusterLoggingTypes.API,
    ClusterLoggingTypes.AUDIT,
    ClusterLoggingTypes.AUTHENTICATOR,
    ClusterLoggingTypes.CONTROLLER_MANAGER,
    ClusterLoggingTypes.SCHEDULER,
  ],
  outputClusterName: true,
  outputConfigCommand: true,
});

// add capacity
cluster.addNodeGroupCapacity('custom-node-group', {
  nodegroupName: 'default-managed',
  minSize: 2,
  maxSize: 20,
  amiType: eks.NodegroupAmiType.AL2_X86_64, // master image
  instanceTypes: [
    new eks.InstanceType('m5.large'),
    // new eks.InstanceType('p3.2xlarge'), // GPU node
  ],
  noeRole: nodeRole, // give this role access to EKS and other AWS services
});

// set up service account
const serviceAccountManifest = cluster.addServiceAccount('eks-admin-service-account', {
  name: 'eks-admin',
  namespace: 'kube-system',
});

const clusterRoleBindingManifest = cluster.addManifest('eks-admin-cluster-role-binding', {
  apiVersion: 'rbac.authorization.k8s.io/v1', // native Kubernetes Role Based Access Control (RBAC)
  kind: 'ClusterRoleBinding',
  metadata: {
    name: 'eks-admin',
  },
  roleRef: {
    apiGroup: 'rbac.authorization.k8s.io',
    kind: 'ClusterRole',
    name: 'cluster-admin',
  },
  subjects: [
    {
      kind: 'ServiceAccount',
      name: 'eks-admin',
      namespace: 'kube-system',
    }
  ],
});

// map a role to system:masters group
cluster.awsAuth.addMastersRole(existingRole);

// Helm charts, e.g. Ray
cluster.addHelmChart('KubeRayOperator', {
  repository: 'https://ray-project.github.io/kuberay-helm/',
  chart: 'kuberay-operator',
  release: 'kuberay-operator',
  version: '1.2.2',
  namespace: ...,
});
...
```

JARK (..., Ray, [[./kubernetes|kubernetes]]) stack in [terraform](https://github.com/awslabs/data-on-eks/tree/main/ai-ml/jark-stack/terraform).

And so on for other components...

## output

```ts
new cdk.CfnOutput(this, 'SecurityGroupId', {
  value: securityGroup.securityGroupId,
});
```

```hcl
output "security_group_id" {
  value = aws_security_group.custom_security_group.id
}
```

## deployment

### AWS CDK

- `cdk list`
- `cdk bootstrap`
- `cdk deploy <stack-component>`
- `cdk destroy --all`

### Terraform

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
