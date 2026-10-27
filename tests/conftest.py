import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ.pop("API_KEY", None)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


GOOD_FLOW = dict(dst_port=443, protocol=6, flow_duration_ms=120.0, fwd_packets=5, bwd_packets=4,
                 fwd_bytes=600, bwd_bytes=2400, pkt_len_mean=333.0, pkt_len_std=120.0,
                 flow_bytes_per_s=25000.0, flow_pkts_per_s=75.0, fwd_iat_mean_ms=30.0,
                 syn_flags=1, ack_flags=8, rst_flags=0, psh_flags=3, init_win_bytes=65535,
                 active_mean_ms=20.0, idle_mean_ms=0.0)

# a classic SYN scan probe: 1 packet, SYN only, tiny window
SCAN_FLOW = dict(GOOD_FLOW, dst_port=31337, flow_duration_ms=0.8, fwd_packets=1, bwd_packets=0,
                 fwd_bytes=52, bwd_bytes=0, pkt_len_mean=52.0, pkt_len_std=2.0, syn_flags=1,
                 ack_flags=0, rst_flags=0, psh_flags=0, init_win_bytes=1024, active_mean_ms=1.0)
