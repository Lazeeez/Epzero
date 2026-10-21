variable "tenancy_ocid" {
  description = "Tenancy OCID (resources are created in the root compartment)."
  type        = string
}

variable "region" {
  type    = string
  default = "ap-tokyo-1"
}

variable "node_count" {
  type    = number
  default = 3
}

# Always Free covers 4 A1 OCPUs and 24 GB in total, and 200 GB of block storage.
variable "ocpus_per_node" {
  type    = number
  default = 1
}

variable "memory_gb_per_node" {
  type    = number
  default = 6
}

variable "boot_volume_gb" {
  type    = number
  default = 50
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/epzero_oci.pub"
}

variable "admin_cidr" {
  description = "Who may SSH in (key-only auth either way)."
  type        = string
  default     = "0.0.0.0/0"
}
