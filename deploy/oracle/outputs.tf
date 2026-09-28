output "public_ips" {
  value = oci_core_instance.node[*].public_ip
}

output "private_ips" {
  value = oci_core_instance.node[*].private_ip
}

output "router_url" {
  description = "HTTPS endpoint of the router (node 1), via a wildcard DNS name for its public IP."
  value       = "https://oryx-${replace(oci_core_instance.node[0].public_ip, ".", "-")}.sslip.io"
}
