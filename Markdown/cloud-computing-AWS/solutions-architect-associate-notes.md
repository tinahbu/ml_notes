@: aws_solutions_architect_associate_acloudguru_notes

# AWS Solutions Architect Associate

## 1. Identify Access Management (IAM)

### 1.1 IAM

- Manage users and their level of access to AWS
- centralized control
  - universal, doesn't apply to region
- shared access to your AWS account
- granular permissions
- Identity Federation
  - Active Directory: log in with Windows login password
  - Facebook / LinkedIn / ...
  - Using SAML (Security Assertion Markup Language 2.0), you can give your federated users single sign-on (SSO) access to the AWS Management Console.
- Multifactor authentication
- Key terminology
  - users
    - root account
      - account when first setup AWS account
      - full admin access
      - always setup Multifactor authentication
    - no users have no permissions when created (least privilege)
    - no users are assigned access key ID and secret access keys when created
      - can't log in to console
      - can access AWS via APIs and command line
  - groups
    - a collection of users
    - An IAM group is used to collectively manage users who need the same set of permissions. By having groups, it becomes easier to manage permissions. So if you change the permissions on the group scale, it will affect all the users in that group. 
  - policies: policy documents in JSON, give permissions as to what a user/role/group can do
  - roles: create roles and assign them to AWS resources 
    - EC2 write to S3
    - can be assigned to EC2 after creation
    - universal
    - You can easily add tags which define which instances are production and which are development instances and then ensure these tags are used when controlling access via an IAM policy.

## 2. Storage

### 2.1 S3 (Simple Storage Service)

- object-based storage

  - key: name
  - value: data (files)
    - 0 byte to 5 TB
    - unlimited storage
    - stored in buckets (folder)
    - all newly created buckets are private by default
    - if uploaded successful, HTTP 200 code
    - can setup logs
  - version ID
  - metadata
  - subresources
    - Access Control List
    - Torrent

- universal name space

  - bucket name needs to be unique globally
  - https://s3-eu-west-1.amazonaws.com/bucket_name

- data consistency

  - read after write consistency: PUTS of new object
  - eventual consistency: overwrite PUTS and DELETES

- guarantee

  - 11 x 9s durability (won't be lost)
  
- Tiered Storage

  | S3 Type                  | Availability | Durability | Features                                                     | Retrieval Time |
  | ------------------------ | ------------ | ---------- | ------------------------------------------------------------ | -------------- |
  | S3 standard              | 99.99%       | 11 * 9     | stored redundantly across multiple device in multiple facilities; sustain loss of 2 facilities concurrently | ms             |
  | S3 - IA                  | 99.9%        | 11 * 9     | rapid but infrequent access; charged with retrieval fee      | ms             |
  | S3 One Zone - IA         | 99.5%        | 11 * 9     | low cost, infrequently access; no multi-AZ                   | ms             |
  | S3 - Intelligent Tiering |              |            |                                                              | ms             |
  | S3 Glacier               |              | 11 * 9     | data archive; low cost                                       | min ~ hr       |
  | S3 Glacier Deep Archive  |              |            | lowest cost                                                  | 12 hours       |
  
- MFA delete
- secure data with
  - access control list
  - bucket policies
  - pre-signed URLs

- Pricing

  - storage

  - requests

  - storage management  (tiers)

  - data transfer

  - transfer acceleration

    - data transfer between user and S3 bucket

    - use CloudFront's edge locations

    - amazon backbone network

      <img src ='s3_data_transfer.png' width=600 align="left">

  - cross region replication

    - sync a bucket to another region
    - versioning must be enabled on both source and destination
    - regions must be unique
    - files in an existing bucket are not replicated automatically 
    - all subsequent updated files will be replicated automatically
    - Delete markers are not replicated
    - deleting individual versions or markers will not be replicated

- encryption

  - default: in stransit
    - SSL/TLS: HTTPS
  - at rest
    - server side encryption
      - S3 managed keys - SSE-S3
      - AWS key management service, SSE-KMS
      - with customer provided keys - SSE-C
    - client side encryption
      - encrypt data -> upload to S3
      - client library such as Amazon S3 Encryption Client

- version control

  - billed for total size

- Lifecycle management

  - automates moving your objects between the storage tiers
  - can be used in conjunction with versioning
  - can be applied to current or previous versions

### 2.2 CloudFront

- Edge location
  - where content is cached 
  - separate to region or AZ
  - not READ only
  - cached for TTL (time to live) in seconds
  - can clear cached objects with charge
- Origin
  - origin of the files that CDN will distribute
    - S3 bucket
    - EC2 instance
    - ELB
    - Route53
  - origin access identity
    - You can optionally secure the content in your Amazon S3 bucket so users can access it through CloudFront but cannot access it directly by using Amazon S3 URLs. 
    - This prevents anyone from bypassing CloudFront and using the Amazon S3 URL to get content that you want to restrict access to. 
- Distribution
  - name given to the CDN with a collection of edge locations
- Web distribution
  - used for websites
- RTMP
  - used for media streaming

### 2.3 Storage Gateway

- concept
  - replicate data from on-premise data center to AWS
- Types
  - File Gateway (NFS): flat file -> objects on S3
  - Volume Gateway (iSCSI protocol)
    - disk volumes -> EBS snapshots on S3
    - stored volume: all data stored in data center (on site) and synced on AWS (async backup)
    - cached volume: most recent used data cached in data center, all data on S3
  - Tape Gateway: backup physical tapes

### 2.4 Snowball

- big disk to move data in and out AWS 
- can be imported to S3
- can export things out of S3

## 3. EC2

### 3.1 EC2 101

- types of instances
  - on-demand: fixed rate with no commitment
  - reserved
    - contract 1 year / 3 years
    - discounted price
    - can't change region
  - spot: bid price
  - dedicated hosts
- Instance - FIGHT DR MC PXZAU
  - F: FPGA
  - I: IOPS
  - G: Graphics
  - H: High disk throughput
  - T: cheap general (t2 micro)
  - D: density
  - R: RAM
  - M: main choice
  - C: compute
  - P: graphics (pics)
  - X: extreme memory
  - Z: extreme memory & CPU
  - A: AIM based workloads
  - U: bare metal
- 1 subnet == 1 availability zone
- Bootstrap Script
  - automating software installs and updates
- to get information about an instance 
  - `curl https://169.254.169.254/latest/meta-data/`
  - `curl https://169.254.169.254/latest/user-data/` bootstrap scripts
- Termination protection
  - prevent your instance from being accidentally terminated from the console, CLI, or API. 
  - disabled by default
  - The `DisableApiTermination` attribute does not prevent you from terminating an instance by initiating shutdown from the instance (using an operating system command for system shutdown) when the `InstanceInitiatedShutdownBehavior` attribute is set.

### 3.2 Security Group

- Updated rule take effect immediately
- Inbound
  - everything blocked by default
  - white list
    - allow rules only, no deny rules
    - use Network Access Control Lists to block specific IP
- Outbound
  - all outbound traffic is allowed
- Stateful
  - If you create an inbound rule allowing traffic in, that traffic is automatically allowed back out again
- EC2: Security Groups = m:m 

### 3.3 Block Storage

- With *block storage*, files are split into evenly sized blocks of data, each with its own address but with no additional information (metadata) to provide more context for what that block of data is.
- Another key difference is that block storage can be directly accessed by the operating system as a mounted drive volume, while object storage cannot do so without significant degradation to performance. 

- EC2 block storage
  - EBS volume
  - instance store volume

### 3.4 EBS Volume

- Concept
  - virtual hard disk drive
    - general purpose SSD: most work loads (gp2)
    - provisioned IOPS SSD: high performance  (DB) (io1)
    - throughput optimized HDD: frequent access, throughput intensive (Big data/data warehouse) (st1)
    - cold HDD: less frequently accessed workloads, lowest cost (sc1)
    - EBS Magnetic: previous generation HDD
  - block based storage
  - volume size can be changed on the fly
    - size 
    - storage type
  - always same availability zone as the EC2 instance
- Termination
  - instance terminated
    - root EBS volume 
      - deleted by default
      - can turn termination protection on
    - additional ECS volumes persists
- Encryption
  - can't encrypt EBS root volume of default AMI
    - create a snapshot of root volume
    - create a copy of snapshot and select encrypt
    - create an AMI from the encrypted copy
    - use the AMI to launch a new instance
  - can encrypt additional volume
- Snapshot
  - exist on S3
  - time copies of the volumes
  - Incremental
  - can be taken when instance is running, but stop the instance first if the EBS serves as root device to ensure consistency
  - Encryption 
    - snapshots of encrypted volumes are encrypted automatically
    - volumes restored from encrypted snapshots are encrypted automatically
    - you can share snapshots, but only if they are unencrypted
      - with other AWS accounts
      - public, sell in marketplace
  - can be used to move EC2 volume to another availability zone
- to move EC2 to another AZ
  - take a snapshot
  - create an AMI from the snapshot
  - use the AMI to launch a EC2 in a new AZ
- to move EC2 to another region
  - take a snapshot
  - create an AMI from the snapshot
  - copy AMI to another region
  - use the AMI to launch a EC2 in the new region
- Can create AMI (Amazon Machine Image) from EBS-backed instances and snapshots
- EBS vs Instance Store
  - instance store (ephemeral storage)
    - The data in an instance store persists only during the lifetime of its associated instance
    - can't be stopped, can reboot or terminate
      - will lose data if stopped, or underlying disk drive fails or instance terminates
      - won't lose data if reboot (intentionally or unintentionally)
      - ROOT volumes will be deleted on termination
    - You can attach additional instance store volumes to your instance. You can also attach additional EBS volumes after launching an instance, but not instance store volumes.
    - the root device for an instance luanched from the AMI is a instance store created from a template stored in S3 (can take a little longer than EBS backed volumes)
  - EBS backed volumes
    - the root device for an instance luanched from the AMI is a EBS volume created from a EBS snapshot
    - can be stopped and snapshotted
    - won't lose data if stopped
    - won't lose data if reboot
    - by default, ROOT volumes will be deleted on termination, but can change setting to keep it

### 3.4 CloudWatch

- monitoring performance
  - CloudTrail is about auditing API calls
- AWS services and user applications
- Interval
  - every 5 minutes by default
  - can have 1 minute interval if turn on detailed monitoring
- Features
  - Alarms
  - Dashboards
  - Events
  - Logs
- Cloud Watch vs Access Logs
  - access logs: requests or connections, the time it was received, the client's IP address, latencies, request paths, and server responses

### 3.5 Command Line

- need to setup access in IAM
- access key ID and secret access key

### 3.6 EFS - Elastic File System

- supports the Network File System version 4 (NSFv4) protocol
- only pay for the storage you use, no pre-provisioning
- can scale to TB
- can support thousands of concurrent NFS connections
  - can't share EBS with multiple EC2
  - can create EFS mount
- stored multi AZ within a region
- read after write consistency

### 3.7 Spread placement groups

- configures the placement of a group of instances

    | type | cluster | spread placement group |
    |:--|:--|:--|
    | definition | packs instances close together inside 1 AZ | spread out across underlying hardware to minimize correlated failures by default |
    | feature | low-latency network performance, high throughput |  |
    | use case | tightly-coupled node-to-node communication that is typical of HPC applications | critical instances | 

- unique namespace within AWS account
- can't merge placement groups
- can't move existing instance to a placement group
- maximum of 7 running instances per Availability Zone

## 4. DB

### 4.1 RDS (OLTP)

- runs on VM
  - but you can't log into these VMs
  - no OS level access
  - AWS patches the OS and DB
  - NOT serverless
  - Aurora Serverless IS serverless
- 6 Engines
    - MySQL
    - PostgreSQL
    - Oracle
    - SQL Server
    - MariaDB
    - Aurora
- Instance Types
    - optimized for memory
    - optimized for performance  
    - optimized for I/O
- Storage
    - general purpose SSD
    - provisioned IOPS SSD: for I/O-intensive workloads, particularly database workloads
    - magnetic: ackward compatibility
    - You can create MySQL, MariaDB, and PostgreSQL RDS DB instances with up to 32 TiB of storage. You can create Oracle RDS DB instances with up to 64 TiB of storage. You can create SQL Server RDS DB instances with up to 16 TiB of storage. For this amount of storage, use the Provisioned IOPS SSD and General Purpose SSD storage types.
- 2 types of Backups
  - automated backups
    - enabled by default
    - Recover to any point in time within a retention period
    - down to a second
    - 1 ~ 35 days
    - daily snapshot + transaction logs throughout the day
    - backup data stored in S3
    - free storage space equal to the size of your DB
    - deleted after deletion of original RDS instance
  - database snapshots
    - done manually / user initiated
    - stored even after deletion of original RDS instance
  - restoring
    - New RDS instance with a new DNS endpoint either from automated backups or snapshots
  - single-AZ RDS instance during snapshot or backup period
    - I/O may be briefly suspended while the backup process initializes (typically under a few seconds), and you may experience a brief period of elevated latency.
  - Change backup window
    - take effect immediately
- Encryption
  - supporting all 6 engines
  - KMS service (Key Management Service)
  - data stored, automated backups, read replicas and snapshots are also going to be encrypted
- Multi-AZ
    - When you provision a Multi-AZ DB Instance, Amazon RDS automatically creates a primary DB Instance and synchronously replicates the data to a standby instance in a different Availability Zone (AZ)
    - for disastor recovery only
        - synchronous replications
        - not for improving performance
        - can't use secondary DB instances for writing purposes
    - exact copy of your DB in another AZ
        - auto replicated
    - auto failover
        - Reboot from failover: force change AZ 
        - Loss of availability in primary Availability Zone 
        - Loss of network connectivity to primary Compute unit 
        - failure on primary Storage 
        - failure on primary 
    - same DNS routed to secondary DB
        - CNAME changed to standby DB
    - supporting all 5 other engines besides Aurora
        - Aurora is fault tolerant itself so no need for multi-AZ
- Read replica
    - performance, scaling
    - must have automatic backups turned on
    - read-only copy of your production DB
        - up to 5
        - allow read replicas of read replicas
        - each read replica has its own DNS end point
        - can have multi-AZ
        - can be in different regions
        - can be created for multi-AZ DB
        - can have a read replica in a second region of your primary DB
        - can be promoted to master, breaks the replication though
        - can be Aurora or MySQL
    - asynchronous replications
    - read from different DB but only write to 1 primary
        - read directed first to primary DB and then sent to secondary DB
    - no auto failover
    - supporting
        - MySQL Server
        - PostgreSQL
        - MariaDB 
        - Aurora
- no charge for replicating data from your primary RDS instance to your secondary RDS instance
- Security Group
    - Technically a destination port number is needed, however with a DB security group the RDS instance port number is automatically applied to the RDS DB Security Group.
- RDS Reserved instances
    - reserve a DB instance for a one or three year term
    - discount compared to the On-Demand Instance pricing
    - available for multi-AZ deployments
- database on EC2
    - EBS

### 4. 2 No-SQL - DynamoDB

- Data Model
  - document 
  - Key-value
- Stored on SSD storage -> fast
- Spread across 3 geographically distinct data centers
- partitioned into nodes - sharding
- Consistency
  - default: eventual consistent reads -> consistency across all copies usually reach within 1 sec 
  - strongly consistent read available -> needs to read an update within 1 sec
- Billing
  - within a single region
    - read and write capacity
    - incoming data transfer
  - cross regions
    - data transfer
- Index
  - global secondary index
    - partition key - different
    - sort key - different
  - local secondary index
    - partition key - same as table
    - sort key - different
    - can only be created when table is created

### 4.3 Warehousing - RedShift (OLAP)

- OLAP
    - business inteligence 
    - PB scale
- different architecture of DB and infrastructure layer
    - single Node (160 GB)
    - multi-node
        - leader node
            - manages clients connections and receives queries
        - compute node
            - stores data and perform queries 
            - up to 128 
- Advanced Compression
    - compression based on columns instead of rows
        - data of same type
        - stored sequentially on disk
    - doesn't require indexes or materialized views
        - use less space
    - auto samples data and selects most appropriate compression scheme when loading data
    - massively parallel processing (MPP)
- Backups
    - enabled by default with 1 day retention period
        - Max retention period 35 days
    - always attempts to maintain at least 3 copies of your data 
        - original on compute node
        - replica on compute node
        - backup in S3
    - asynchronously replicate your snapshots to S3 in another region for disaster recovery
- Billing
    - billed for compute node hours
        - 3 node cluster run for 1 day will incur 72 billable hours
        - only compute node, not leader node
    - billed for backup
    - billed for data transfer (only within a VPC, not outside it)
- Security
    - encrypted in transit using SSL
    - encrypted at rest using AES-256 
    - by default redshift takes care of key management
        - can manage your own key through 
            - HSM
            - KMS key management service
- Availability
    - currently only available in 1 AZ
    - can restore snapshots to new AZ in the event of an outage

### 4.4 Aurora

- MySQL & PostgreSQL compatible 
- open source
- cost effective
- Scalility
  - auto scaling storage
  - start with 10 GB, scales in 10GB increments to 64 TB
  - compute resources: scale up to 32vCPUs and 244GB of memory
- Availability
  - 2 copies of your data is contained in each AZ
  - minimum of 3 AZ 
  - -> 6 copies of your data
  - designed to handle the loss of 
    - up to 2 copies of data without affecting DB write availability
    - up to 3 copies without affecting read availability
  - Self-healing
  - automated backups always enabled 
    - doens't impact performance
  - snapshots 
    - doesn't impact performance
    - can be shared with other AWS accounts
- Performance
  - read replicas 
    - Aurora replicas: up to 15
    - MySQL replicas: up to 5
  - Automated failover from Aurora to Aurora read replica
  - No Automated failover from Aurora to MySQL read replica 
- Endpoints
  - cluster: primary instance for write operations
  - reader: read replicas
  - instance: individual instance

### 4.5 ElastiCache

- in-memory caching
  - Memcached
    - simple cache to offload DB
    - scale horizontally
    - multi-threaded performance
  - Redis
    - advanced data types
    - Ranking/sorting
    - pub/sub
    - persistence
    - multi-AZ
    - Backup & restore capabilities
- speed up existing DB (caching frequent identify queries)
  - improves web app performance
    - in-memory cache instead of disk-based DB

## 5. Route53

### 5.1 DNS 101

- Naming
  - The "route" part comes from the historic "Route 66" of the USA
  - 53 is port 53 for TCP and UDP
- DNS
  - convert human friendly domain names -> Internet Protocol (IP) address
  - IP addresses are used by computers to identify each other on the network
  - IPv4 or IPv6 
    - IPv4
      - 32 bit field
      - 4 billion addresses
      - running out
    - IPv6
      - 128 bits
      - 340 undecillion addresses
- top level domains
  - string of characters seperated by dots
  - last word is top level domain
    - .com
    - .edu
    - .gov 
  - second level domain name
    - .co.uk
      - .uk is top level
      - .co is second level domain
  - Controlled by the Internet Assigned Numbers Authority (IANA)
  - domain registrars
    - assign domain names directly under 1 or more top-level domains
    - Amazon
    - GoDaddy.com
    - 123-reg.co.uk etc.
  - SOA record
    - start of authority record
  - NS record
    - name server record
    - used by top level domain servers to direct traffic to the content DNS server
    - "acloudguru.com" -> .com server -> NS records -> SOA (DNS records)
  - DNS record
    - A record
      - A = address
      - domain name -> IP address
    - TTL
      - time to live
      - length a DNS record is cached in the resolving server or user local PC
      - lower TTL, faster changes to DNS records take to propagate throughout the internet
    - CName
      - Canonical Name
      - resolve 1 domain name to another
      - "m.acloudguru.com" -> "mobile.acloudguru.com" -> "IPv4"
      - BATMAN -> "Bruce Wayne" -> 412-412-4121
    - Alias Records
      - map resource records to ELB, CloudFront, or S3 buckets websites
      - DNS -> target name
      - can't be used for naked domain name (without a www.)
      - has to be an A record or an alias
      - always choose an alias record over a CName in an exam scenario
      - Provides a router53-specific extension to DNS functionality
  - MX record
  - PTR record
    - reverse of an A Record
    - address -> domain name
  - ELBs don't have pre-defined IPv4 addresses, you resolve to them using a DNS name

### 5.2 Route53

- Domain Registration
  - can buy domain names directly with AWS
  - take up to 3 days to register
- Create 3 EC2 in 3 regions
  - N. Virginia 3.87.24.95 
  - Seoul 54.180.113.112
  - Paris 35.181.48.173
- Routing policies
  - simple routing
    - 1 record with multiple IP address
    - Route53 returns a random value
    - hosted zones -> create record set -> naked name -> A record -> put 3 IPs for EC2 in
    - access tinabu.com and the server will change
    - can modify TTL
  - weighted routing
    - split traffic based on weights
    - e.g. 20% Paris 80% N. Virginia
  - latency-based routing 
    - route your traffic based on the lowest network latency for your end user
    - fastest response time
  - failover routing
    - create an active/passive set up
    - e.g. primary site in EU-WEST-2 and secondary disaster recovery site in AP-SOUTHEAST-2
    - failover to passive if active is down
  - geolocation routing
    - choose where your traffic will be sent based on the user's geographic locations
    - continent / country based
  - geoproximity routing
    - let Route53 route traffic based on user geographic location and your resources
    - must use Route53 traffic flow
    - you can influence it with "bias"
    - traffic policy tab
  - multivalue answer routing
    - like simple routing: configure Route53 to return multiple values 
    - but plus health checks and return only healthy resources
- Health check
  - create on individual record sets
  - Add health check to A record 
  - if a record fails a health check, it will be removed from Route53 until it passes the health check again
  - can set SNS notificatin for health check failures

## 6. VPC

5~10+ questions

### 6.1 Virtual Private Cloud Basics

<img src ='vpc_diagram.png' width=500>

- logically solated section (data center) of AWS cloud 
  
- Your VPC must have at least one subnet in at least two of the Availability Zones in the region
  
- complete control over the environment
  
  - select IP address
    - assign custom IP address ranges in each subnet
  - create subnets (subnet: AZ M:1)
    - not created automatically when VPC created
    - launch instances into a subnet of your choice
    - public-facing
      - web servers
      - jump boxes/bastion host: EC2 in public subnet that is used to connect to an instance in private subnet
    - private-facing
      - backend systems 
        - DB
        - application servers
      - no internet access
      - IP ranges reserved for private network
        - 10.0.0.0 - 10.255.255.255 (10/8 prefix) AWS doesn't allow /8 
        - 172.16.0.0 - 172.31.255.255 (172.16/12 prefix)
        - 192.168.0.0 - 192.168.255.255 (192.168/16 prefix)
  - configure route tables between subnets
    - created automatically when VPC created
  - create network gateways and attach to VPC
    - not created automatically when VPC created
    - only 1 internet gateway per VPC
  - much better security control over AWS resources
  
- can create a Hardward Virtual Private Network (VPN) connection between corporate data center and a VPC
  
  <img src='vpc_vpn.png' width=250>
  
- default VPC
  - all subnets in default VPC have a route out to the internet
  - each EC2 instance has both a public and private IP address
  - in VPC, an instance retains its private IP.

- VPC peering
  - connect 1 VPC with another via a direct network route using private IP addresses

  - instances behave as if they were on the same private network

  - can peer VPC with other AWS accounts

  - star configuration

    - 1 central VPC peers with 4 others
    - no transitive peering
    - must set up new peering relationship

    <img src ='vpc_peering.png' width=200>

  - peer between regions

## 6.2 Create a Customer VPC

- IP range: 10.0.0.0/16
- automatically created
  - route table
  - network ACL
  - security group
- won't create
  - subnet
  - internet gateway
- us-east-1a for you can be completely different than us-east-1a for someone else
- Create subnets
  - 5 addresses are reserved by AWS
    - network, router, DNS server, network broadcast address
  - enable auto assign public IP for 1 instance
- Create gateway
  - attach to VPC
  - only 1 gateway attached to 1 VPC
- configure main route
  - default 2 routes
    - allows talk to each other over IPv4 and IPv6
  - Main route table
    - needs to be as private as possible
  - create a public route table
    - any instance associated with it is able to talk to the internet
    - add routes
    - destination: 0.0.0.0/0, target: internet gateway
    - Destination: ::/0 (IPv6), target: internet gateway
    - subnet association
      - pick the public subnet to associate

### 6.3 NAT Instances & Gateways

- concept
  - Network Address Translation
  - enable EC2 in private subnets to communicate to internet or other AWS services
  - but prevent the internet from initiating a connection
- NAT Instance 
  - single EC2 instance (Community AMI available)
  - change Source/Dest. Check to disable it
  - must be in a public subnet
  - main route table: add route 0.0.0.0/0 NAT instance
  - must be a route out of the private subnet to the NAT instance
  - could be a bottle neck
    - amount of traffic supports depends on the instance size
  - behind a security group 
- NAT Gateway
  - Create a NAT gateway
    - create from public subnet
    - specify an elastic ip address to associate with the NAT gateway
    - edit route table: add private subnects
    - main route table: add route 0.0.0.0/0 NAT gateway
    - not associated with security group
    - auto assigned a public IP address
    - no need to disable source/dest. check
  - redundant inside the availability zone
    - more available and scalable
    - can't span AZ
    - scale automatically
    - no need to patch
  - create an NAT gateway in different regions to allow high availability

### 6.4 Network Access Control List vs Security Group

- ACL
  - created by default
    - all subnets associated with it automatically
    - default allows all outbound and inbound traffic
    - subnet:ALC M:1
  - create custom ACL
    - deny all inbound and outbound by default
    - edit subnet associations: add public subnet
    - add inbound rules
    - add outbound rules
  - ephemeral port
    - short-lived transport protocol port for IP communications
  - Evaluate the rules chronological order
    - DENY before ALLOW
    - handle by change the rule#
  - rule change take effect immediately
- stateless
  
- instance security group
  - second line of defense
  - stateful 
  - created automatically when VPC created
  - can't span VPCs
  - Security Groups evaluate all rules before deciding whether to allow traffic
- subnet network access control list (ACL)
  - first line of defense
  - stateless
  - allow rules
  - deny rules
  - separate inbound and outbound rules
  - opened for inbound doesn't automatically open it for outbound 
  - created automatically when VPC created

### 6.5 Custome ACL and ELB

- create an ELB with 2 subnets (ELB can only have 1 region)
- at least 2 public subnets needed

### 6.6 VPC Flow Logs

- a feature that enables you to capture information about the IP traffic going to and from network interfaces in your VPC
- data stored using CloudWatch Logs / S3
- created at 3 levels
  - VPC
  - subnet
  - network interface level 
- VPC -> Actions -> Create flow log -> set up permissions
- cannot enable flow logs for VPC that are peered with VPC unless the peer VPC is in your account
- can't tag a flow log
- after creating a flow log, can't change its configuration (associate a different IAM role)
- not all IP traffic is monitored
  - traffic to and from 169.254.169.254 is for metadata

### 6.7 Bastion Host

- securely administer EC2 instances (with SSH/RDP)
- can use a NAT Gateway as a Bastion host
- special purpose computer designed to withstand attacks
- in public subnet
- SSH/RDP bastion instance forward traffic to private instance
- Hardening this because this is going to be hacked

### 6.8 Direct Connect

- a cloud service solution to establish a dedicated network connection from your premises (data center, office) to AWS

- high throughput workloads (a lot of network traffic)

- stable and reliable secure connection

  <img src ='vpc_direct_connect.png' width=600>

### 6.9 VPC Endpoints

<img src ='vpc_endpoint_before.png' width=600>

<img src ='vpc_endpoint_after.png' width=600>

- enables private connection to your AWS services from VPC
- using PrivateLink without internet gateway, NAT device, VPN or Direct Connect
- instances in yoru VPC don't require public IP to communicate with resources in the service
- traffic between VPC and other services don't go over the internet, doesn't leave the Amazon network
- 2 types
  - interface endpoint
  - gateway endpoint
    - S3
    - Dynamo DB
    - Add S3 admin role to private instance
    - create endpoint for S3 main route table
    - will see update in route table in a few minutes
- horizontally scaled, redundant

## 7. HA

### 7.1  Load Balancers

~10 questions

- Balance load across web servers
  - Application load balancer
    - intelligent
    - advanced request routing
    - layer 7 - application aware
  - network load balancer
    - TCP traffic
    - extreme performance FAST FAST FAST 
    - layer 4 - connection level
    - designed for high performance traffic that is not conventional Web traffic.
  - classic load balancer / elastic load balancer
    - low cost
    - HTTP/HTTPS applications (layer 7: x-forwarded and sticky sessions, layer 4: applications TCP protocol)
    - X-forwarded-for header: pass the public IPv4 address of end user to the load balancer
- 504 Error
  - gateway timed out
  - application not responding
  - trouble shoot the application
    - web server / db
    - scale it up or out if necessary
- DNS address instead of IP
  - IP can change
  - but you can get an elastic IP address for network load abalancer 
- Health Check
  - load balancer will ping the instance to check its health
  - InService / OutOfService

- Sticky sessions

- Cross Zone load balancing

  <img src ='ha_cross_zone_load_balancer.png' width=500>

- Path patterns

  - direct traffic to different EC2 based on the URL contained in the request

  <img src ='ha_path_patterns.png' width=500>

### 7.2 Design a highly available wordpress blog site

- Create S3 buckets: 1 code 1 media files

- Create Cloud Front distribution with media bucket

- Add web security group (HTTP port 80 and SSH port 22) to the inbound rules of DB security group (port 3306 for DB) 

- Create a MySQL instance with VPC DB security group

- IAM create a role for EC2 to speak to S3

- Create a EC2 

- add RDS endpoint as DB host in Wordpress setup page

- copy the script to wp-config.php, then run the installation

- createa new post with images

- in folder wp-content/uploads/year/month there will be the images

- every time an image is uploaded, stored in S3 for redundancy

  ```shell
  aws s3 cp --recursive /var/www/html/wp-content/uploads s3://media_bucket_name
  ```

- eventually force wordpress to serve from CloudFront distribution other than image on EC2 for better performance

- full copy of wordpress site to another S3 bucket

  ```shell
  aws s3 cp --recursive /var/www/html/ s3://code_bucket_name
  ```

- in .htaccess update domain name to cloud front 

- sync EC2 data to S3

  ```shell
  aws s3 sync /var/www/html s3://code_bucket_name
  ```

- allow URL rewrite rules to use CloudFront instead of public IP

  ```
  cd /etc/httpd/conf
  cp httpd.conf httpd-copy.conf
  # AllowOverride controls change from None to All
  ```

- make S3 public by modifying Bucket policy

  ```json
  {
    "Version": "2019-04-17",
    "Statement": [
      {
        "Sid": "PublicReadGetObject",
        "Effect": "Allow",
        "Principal": "*",
        "Action": [
          "s3:GetObject"
          ],
        "Resource": [
          "arn:aws:s3:::BUCKET_NAME/*"
          ]
      }
    ]
  }
  ```

- Create a application load balancer with new target group

  - put EC2 into the target group

- Route53 point our domain name to the load balancer

  - Add A record with load balancer Alias 

- Configure a reader node

  <img src ='ha_wordpress_autoscal.png' width=300>

  - create a reader node template

  - configure current instance to scan S3 every minute, download changes to our site

    ```
    */1 * * * * root aws s3 sync --delete s3://code_bucket_name /var/www/html
    # run command */1 every minute
    # every hour of every day of the month of every day of the week
    ```

  - Create an image of this instance as reader node template

  - go to EC2 and change it to primary write node that syncs its files to S3 buckets

    ```
    */1 * * * * root aws s3 sync --delete /var/www/html s3://code_bucket_name
    */1 * * * * root aws s3 sync --delete /var/www/html/wp-content/uploads/ s3://media_bucket_name
    ```

  - reader node: prevent general public from accessing this node

- Create auto scaling group

  - create a new launch configuration

    - Use reader AMI created
    - choose the S3 IAM role
    - Add bootstrap script

    ```shell
    #!/bin/bash
    yum update -y
    aws s3 sync --delete s3://code_bucket_name /var/www/html
    ```

    - use webDMZ security group

  - Create auto scaling group

    - remove write node from target group
    - 2 instances
    - Add target group advanced details 
    - ELB health check 
    - 2 reader nodes will be created 

### 7.3 CloudFormation

- a way to completely scripting your cloud environment
- across all regions and accounts
  - create a stack
  - from sample template - WordPress blog
  - a EC2 with wordpress and MySQL will be created
- Quick Starts: a bunch of Cloud Formation templates built, allowing users to build complex environment quickly
- Components
  - Parameters
  - Resources
  - Outputs

### 7.4 Elastic Beanstalk

- under Compute services
- Easy application deployment with code upload and clicks 
- Configures EC2 provisioning, load balancing, scaling, and health check
- can configure auto scaling
- good for web applications

## 8. Application

### 8.1 SQS - Simple Queue Service

- Concept
  - message queue that stores messages while waiting to be processed
  - decouple infrastructure
  - buffer between components
- message
  - default 256 KB of text
  - up to 2GB (stored in S3 instead)
  - persistence
    - 1 min to 14 days
    - default retention period 4 days
  - visibility time out
    - amount of time the message is invisible in the SQS after a reader picks up that msg
    - deleted if message is processed after visibility timeout expires
    - visible again if message is not processed so in case the first EC2 died, another one can pick it up after the time out
    - may result in a message being delivered twice
    - max 12 hours
- pull based
  - short polling: returns immediately
  - long polling: doesn't return until a message arrives in the queue or the long poll times out
- 2 types
  - standard queue
    - unlimited number of transactions per second
    - at least once
    - occasionally more than 1 copy of a message might be delivered out of order
    - generally delivered in same order as they are sent
  - FIFO queue
    - FIFO: order strictly preserved
    - exactly once: no duplicates
    - 300 transactions per second (TPS)

### 8.2 SWF - Simple Work Flow 

- concept
  - coordinate work across distributed application components
- tasks
  - executable code, code service calls, human actions, scripts
  - e.g. Amazon warehousing (code: process transaction, human: pick up inventory)
- Actors
  - Workflow starters: initiate a workflow
  - Deciders: control the flow of tasks execution
  - Activity workers: carry out the tasks
- domain: collection of workflows
- SWF vs SQS
  - SQS
    - retention period of up to 14 days
    - message-oriented API
    - may have duplicated message 
    - Need to implement your own application-level tracking
  - SWF
    - workflow executions can last up to 1 year
    - task-oriented API
    - task is ensured to be assigned only once and never duplicated
    - tracks all tasks and events in an application

### 8.3 SNS - Simple Notification Service

- concept
  - push notifications (push based)
  - multiple transport protocol
    - mobile: Apple, Google, Fire OS, Windows devices, Android
    - SMS text messages
    - email to SQS queues
    - HTTP endpoint 
- group multiple recipient using topics
  - topics: access point for subscribers
  - 1 topic - M endpoint types
- stored redundantly across multiple AZ

### 8.4 Elastic Transcoder

- concept
  - media transcoder 
  - convert media files to different formats
    - smartphones, tablets, PCs
- pricing
  - minutes you transcode and resolution

### 8.5 API Gateway

5 ~ 10 questions

- Features
  - publish, maintain, monitor and secure API
  - doorway into your AWS environment
  - front door for applications to access data, business logic, or backend services
  - endpoints
    - hostname of the API
    - HTTPS for RESTful API
    - edge-optimized: geographically distributed clients, access across regions
    - regional: clients in the same region
    - private: only accessible from VPC
  - serverless connection to Lambda & DynamoDB
  - send each API endpoint to a different target
  - scales automatically
  - can throttle API gateway to prevent attacks
  - maintain versions of API
- Define an API (container)
- Define resources and nested resources (URL paths)
- for each resource
  - select supported HTTP methods
  - set security
  - choose target: EC2, Lambda, DynamoDB, etc
  - set request and response transformation
- Deploy
  - to a stage
  - with domain name
  - supports certificate manager (SSL/TLS), HTTPS
- Caching
  - caches API gateway endpoints responses
    - reduce number of calls 
    - improve performance
    - TTL in seconds
- Same origin policy
  - 1 page can talk to another only if they have same origin (domain name)
  - to prevent cross site scripting attacks
  - enforced by browser
  - CORS (cross origin resource sharing)
    - server end (not client end in the browser) can relax the same origin policy
    - solve ERROR "origin policy cannot be read at the remote resource"
    - if using Javascript/AJAX that uses multiple domains
    - enforced by client browser
- API Keys
  - to identify an API client and meter their access
  - put quota limits on client
- logging in CloudWatch
  - Access logging
    - customized
    - log who has accessed your API and how
  - Execution logging 
    - API Gateway manages the CloudWatch Logs
    - errors or execution traces (such as request or response parameter values or payloads), data used by Lambda authorizers (formerly known as custom authorizers), whether API keys are required, whether usage plans are enabled, and so on.

### 8.6 Kinesis

- accepts streaming data
  - generated continuously 
  - by thousands of data sources
  - small sizes Kb
- types
  - Kinesis Streams
    - store streaming data in shards
    - persistent: 24 hours by default, up to 7 days
  - Kinesis Firehose
    - optional lambda function
    - no data persistent
    - output immediately (S3, Elastic Search Cluster)
  - Kinesis Analytics
    - analyze data in Streams/Firehose

### 8.7 Cognito - Web Identity Federation

- Concept

  - AWS web identity federation services (identity broker)
  - give user access with web-based identity provider like Amazon / Facebook / Google 
  - User first authenticates with web ID provider -> authentication token
  - exchange authentication token for AWS credentials -> IAM role
    - sign up and sign in to your apps
    - access for guest users
    - synchronizes user data across devices
      - push synchronization

- User pools

  - user based 
    - user registration
    - authentication
    - account recovery 
    - sign up and sign in functionality (username, password)

- Identity pools

  - provide temporary credentials and authorization to access AWS services

  <img src ='app_cognito.png' width=500 align="left">

### 8.8 EMR - Elastic Map Reduce

- a managed Hadoop framework
- big data use cases:
  - log analysis
  - web indexing
  - data transformations (ETL)
  - machine learning

## 9. Serverless

### 9.1 Lambda

- Some History of Cloud
  - Data Center (physical hardware, assembly code/protocols, high level languages, OS, Application layer) -> 
  - IAAS (AWS EC2 launched in 2006) -> 
  - PAAS (Microsoft, AWS Elastic Beanstalk, only need to upload code without worrying about infrastructure configuration) -> 
  - Containers (2014 Docker) -> 
  - Serverless (Code)
- Lambda
  - handles provisioning and managing the servers
  - event-driven
  - compute service runs to respond to HTTP requests
  - independent
    - 1 event = 1 function
  - Serverless 
  - lambda functions can trigger other lambda functions
    - 1 event = x functions 
  - debug: AWS X-ray
  - can do things globally, can use it to back up S3 buckets
- Traditional vs Serverless Architecture
  - Traditional
    - user -> route53 -> load balancer -> web server -> backend DB server -> response to user
  - Serverless
    - user -> API gateway -> Lambda -> backend DB server -> response to user
    - serverlsess DB: Dynamo DB, Aurora
    - RDS is not serverless except Aurora Serverless
    - no servers!
    - continuous scaling!
      - scales out not up automatically
    - super cheap!
- Languages
  
  - Node.js, Java, Python, C#, Go, Power Shell
- Price
  - number of requests
    - first 1 million requests free
    - $0.20 per 1 million requests thereafter
  - duration
    - time your code is executed 
    - and memory usage
- Triggers
  - API Gateway, IoT, dynamoDB, SQS, Application Load Balancer, Alexa, Kinesis, SNS, CloudWatch, CloudFront, CodeCommit, S3
  - S3 events can trigger
  - RDS can't trigger

- Build a serverless webpage

  ​	<img src ='lambda_architecture.png' width=500>

  - Create a lambda function
  
    - start from scratch
  
  - put in name, select Python 3.6
    
  - create function
    
  - modify function code
    
  ```python
      import json
      
      def lambda_handler(event, context):
        # TODO implement
          return {
              'statusCode': 200,
              'headers': {
                  "Access-Control-Allow-Origin": "*",
              },
              'body': json.dumps('Hello from Tina\'s Lambda function!')
          }
  ```
  
    - Add a trigger 
    
      - API Gateway
      - Create a new API, security: AWS IAM, ADD
    
    - Modify API endpoint
    
      - Click on API name (not the endpoint URL)
      - Delete the ANY method
      - Create a GET method 
      - select Lambda Function, select Lambda Proxy integration, add Lambda function name
      - ACTION -> deploy API with default stage
      - Stages -> GET method -> Invoke URL -> should see message in  `json.dumps()`
    
  - Create a S3 bucket to host a static website
    - create a S3 (same name as your Route53 domain)
  
    - edit public access checking: make it public by uncheck everything so "Objects can be public"
  
    - properties -> turn on static website hosting
  
    - upload `index.html` and `error.html`
  
      - with GET method URL updated in `index.html`
  
        ```html
        xhttp.open("GET", "API gateway GET method invoke URL", true)
        ```
  
    - make the 2 html files public from actions - make public
  
  - Click on Object URL for index.html
  
  - Click on "click me" which triggers our lambda function, should see text change
  
  - optional configure Route53 to link to S3 bucket
  
    - add A record with bucket alias

### 9.2 Alexa Skill

- Different components

  - speech recognition
  - Natural language understanding
  - text to speech
  - skills 
  - learning

- Create an Alexa Skill that calls a Lambda Function

  - Polly

    - Add some plain text
    - Synthesize to S3
      - creating a MP3 file

  - Lambda Function

    - From serverless app repository: choose alexa-skills-kit-nodejs-factskill 
    - deploy

  - Create a S3 bucket

    - edit public access checking: make it public by uncheck everything

    - add to Permissions -> bucket policy 

      ```json
      {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::1bigthing-alexa-skill/*"
          }
        ]
      }
      ```

  - Create an Alexa skill that plays this mp3 file

    - Create a alexa skill from https://developer.amazon.com/alexa/
    - "fact skill"
    - go to endpoint, paste Lambda ARN to Default Region 
    - Add intent
      - Add utterrance 
    - Build model
    - Test