"""IK endpoint tests exercise the same HTTP route as click-to-reach."""
from http.server import ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from arm_viewer import ArmHandler


@pytest.fixture(scope="module")
def ik_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ArmHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/ik"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reachable_target_exposes_two_valid_fk_poses(ik_url):
    with urlopen(ik_url + "?x=1&y=0.75", timeout=3) as response:
        payload = json.load(response)
    assert payload["reachable"] and not payload["singular"]
    assert {s["branch"] for s in payload["solutions"]} == {"elbow-up", "elbow-down"}
    for s in payload["solutions"]:
        assert np.allclose(s["pose"]["positions"][-1], payload["target"], atol=1e-12)
        assert s["error"] < 1e-12


@pytest.mark.parametrize("query,reason", [("x=0&y=0", "inside_hole"),
                                         ("x=3&y=0", "outside_reach")])
def test_unreachable_target_is_a_valid_result_not_an_http_error(ik_url, query, reason):
    with urlopen(ik_url + "?" + query, timeout=3) as response:
        assert response.status == 200
        payload = json.load(response)
    assert not payload["reachable"] and payload["reason"] == reason
    assert payload["solutions"] == []


def test_origin_family_with_equal_links_and_shoulder_hint(ik_url):
    with urlopen(ik_url + "?x=0&y=0&l1=1&l2=1&shoulder_hint=30", timeout=3) as response:
        payload = json.load(response)
    assert payload["infinite"] and payload["singular"]
    s = payload["solutions"][0]
    assert np.isclose(s["pose"]["angles_deg"][0], 30)
    assert s["error"] < 1e-12


def test_outer_boundary_is_reported_as_one_solution(ik_url):
    with urlopen(ik_url + "?x=1.75&y=0", timeout=3) as response:
        payload = json.load(response)
    assert payload["singular"] and not payload["infinite"]
    assert len(payload["solutions"]) == 1


@pytest.mark.parametrize("query", ["x=nan", "y=inf", "x=", "x=abc", "x=1&x=2",
                                  "l1=0", "l2=2", "shoulder_hint=181"])
def test_invalid_ik_inputs_receive_json_errors(ik_url, query):
    with pytest.raises(HTTPError) as error:
        urlopen(ik_url + "?" + query, timeout=3)
    assert error.value.code == 400
    assert "error" in json.load(error.value)
