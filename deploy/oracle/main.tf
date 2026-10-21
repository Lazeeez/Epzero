# Epzero cluster on Oracle Cloud Always Free: three Ampere A1 nodes (1 OCPU / 6 GB each, within the
# free 4 OCPU / 24 GB), one VCN, one public subnet. Only HTTPS (router) and key-only SSH are open to
# the internet; shard-to-shard and router-to-shard traffic stays on the private network.
#
#   terraform init && terraform apply
#
# Credentials come from ~/.oci/config (profile DEFAULT); nothing secret lives in this directory.

terraform {
  required_version = ">= 1.6"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0"
    }
  }
}

provider "oci" {
  config_file_profile = "DEFAULT"
  region              = var.region
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_core_images" "ubuntu" {
  compartment_id           = var.tenancy_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "24.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

locals {
  vcn_cidr    = "10.42.0.0/16"
  subnet_cidr = "10.42.1.0/24"
  image_id    = data.oci_core_images.ubuntu.images[0].id
}

resource "oci_core_vcn" "epzero" {
  compartment_id = var.tenancy_ocid
  cidr_blocks    = [local.vcn_cidr]
  display_name   = "epzero"
  dns_label      = "epzero"
}

resource "oci_core_internet_gateway" "epzero" {
  compartment_id = var.tenancy_ocid
  vcn_id         = oci_core_vcn.epzero.id
  display_name   = "epzero-igw"
}

resource "oci_core_route_table" "public" {
  compartment_id = var.tenancy_ocid
  vcn_id         = oci_core_vcn.epzero.id
  display_name   = "epzero-public"
  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.epzero.id
  }
}

resource "oci_core_security_list" "nodes" {
  compartment_id = var.tenancy_ocid
  vcn_id         = oci_core_vcn.epzero.id
  display_name   = "epzero-nodes"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }
  # Everything inside the VCN: router <-> shards <-> replicas.
  ingress_security_rules {
    source   = local.vcn_cidr
    protocol = "all"
  }
  # SSH (key-only) for deploys.
  ingress_security_rules {
    source   = var.admin_cidr
    protocol = "6"
    tcp_options {
      min = 22
      max = 22
    }
  }
  # HTTP (ACME challenge + redirect) and HTTPS for the public router API.
  dynamic "ingress_security_rules" {
    for_each = [80, 443]
    content {
      source   = "0.0.0.0/0"
      protocol = "6"
      tcp_options {
        min = ingress_security_rules.value
        max = ingress_security_rules.value
      }
    }
  }
  # Path MTU discovery.
  ingress_security_rules {
    source   = "0.0.0.0/0"
    protocol = "1"
    icmp_options {
      type = 3
      code = 4
    }
  }
}

resource "oci_core_subnet" "nodes" {
  compartment_id    = var.tenancy_ocid
  vcn_id            = oci_core_vcn.epzero.id
  cidr_block        = local.subnet_cidr
  display_name      = "epzero-nodes"
  dns_label         = "nodes"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.nodes.id]
}

resource "oci_core_instance" "node" {
  count               = var.node_count
  compartment_id      = var.tenancy_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "epzero-node${count.index + 1}"
  shape               = "VM.Standard.A1.Flex"

  shape_config {
    ocpus         = var.ocpus_per_node
    memory_in_gbs = var.memory_gb_per_node
  }

  source_details {
    source_type             = "image"
    source_id               = local.image_id
    boot_volume_size_in_gbs = var.boot_volume_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.nodes.id
    assign_public_ip = true
    private_ip       = cidrhost(local.subnet_cidr, 11 + count.index) # 10.42.1.11, .12, .13
    hostname_label   = "node${count.index + 1}"
  }

  metadata = {
    ssh_authorized_keys = trimspace(file(pathexpand(var.ssh_public_key_path)))
    user_data           = base64encode(file("${path.module}/cloud-init.yaml"))
  }

  lifecycle {
    ignore_changes = [source_details[0].source_id] # a newer image must not recreate the nodes
  }
}
