import time

from fastapi.testclient import TestClient

from transcription_service.api import app

client = TestClient(app)


def _wait_for(job_id: str, timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/transcriptions/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_async_upload_flow(short_wav):
    with short_wav.open("rb") as f:
        r = client.post("/v1/transcriptions", files={"file": ("meeting.wav", f, "audio/wav")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = _wait_for(job_id)
    assert job["status"] == "completed" and job["progress"] == 100

    result = client.get(f"/v1/transcriptions/{job_id}/result").json()
    assert result["file_name"] == "meeting.wav"
    assert result["transcription"]["segments"]

    srt = client.get(f"/v1/transcriptions/{job_id}/result?format=srt")
    assert srt.headers["content-type"].startswith("application/x-subrip")

    assert client.delete(f"/v1/transcriptions/{job_id}").status_code == 204
    assert client.get(f"/v1/transcriptions/{job_id}").status_code == 404


def test_concurrent_uploads(short_wav):
    ids = []
    for i in range(5):
        with short_wav.open("rb") as f:
            ids.append(client.post("/v1/transcriptions", files={"file": (f"f{i}.wav", f)}).json()["job_id"])
    assert len(set(ids)) == 5
    assert all(_wait_for(j)["status"] == "completed" for j in ids)


def test_sync_endpoint_returns_vtt(short_wav):
    with short_wav.open("rb") as f:
        r = client.post("/v1/transcriptions/sync?format=vtt", files={"file": ("a.wav", f)})
    assert r.status_code == 200 and r.text.startswith("WEBVTT")


def test_unsupported_format_rejected():
    r = client.post("/v1/transcriptions", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 415


def test_corrupt_file_fails_without_retry_flag():
    r = client.post("/v1/transcriptions", files={"file": ("broken.wav", b"garbage-bytes")})
    job = _wait_for(r.json()["job_id"])
    assert job["status"] == "failed" and job["retryable"] is False
    assert client.get(f"/v1/transcriptions/{job['job_id']}/result").status_code == 409


def test_result_before_completion_and_missing_job():
    assert client.get("/v1/transcriptions/does-not-exist/result").status_code == 404
