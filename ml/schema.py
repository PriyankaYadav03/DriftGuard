"""Single source of truth for the flow schema shared by training and serving."""
FEATURES = [
    "dst_port", "protocol", "flow_duration_ms",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "pkt_len_mean", "pkt_len_std",
    "flow_bytes_per_s", "flow_pkts_per_s", "fwd_iat_mean_ms",
    "syn_flags", "ack_flags", "rst_flags", "psh_flags",
    "init_win_bytes", "active_mean_ms", "idle_mean_ms",
]
