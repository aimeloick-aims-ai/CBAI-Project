"""Acquire the fixed 1 GB proposal with resumable FTP, three workers and receipts."""

import csv
import getpass
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from ftplib import FTP, Error

from bracs_data import HOST, RAW, ROOT, digest
from PIL import Image

CAP = 1_000_000_000


def main():
    proposal = ROOT / "reports/bracs_expansion/1GB_proposal.csv"
    with proposal.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if sum(int(r["bytes"]) for r in rows) > CAP:
        raise ValueError("Manifest exceeds the authorized 1 GB")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image")
    user = os.environ.get("BRACS_FTP_USER") or input("FTP username: ")
    password = os.environ.get("BRACS_FTP_PASSWORD") or getpass.getpass("FTP password: ")
    lock = threading.Lock()
    received = [0]
    completed, failures = [], []
    out = ROOT / "reports/bracs_expansion"

    def transfer(row):
        target = (RAW / "RoI" / row["split"] / row["category"] / row["image_id"]).resolve()
        if not target.is_relative_to(RAW.resolve()):
            raise ValueError("Invalid local target")
        size = int(row["bytes"])
        url = f"ftp://{HOST}{row['remote_path']}"
        receipt = target.with_name(target.name + ".receipt.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial_meta = target.with_name(target.name + ".part.meta.json")
        if target.exists():
            record = json.loads(receipt.read_text())
            if (
                record["source_url"] != url
                or record["sha256"] != digest(target)
                or target.stat().st_size != size
            ):
                raise ValueError("Cached file mismatch")
        else:
            if partial.exists() and partial.stat().st_size and not partial_meta.exists():
                raise ValueError("Untracked partial file; explicit review required")
            metadata = {"url": url, "bytes": size}
            if partial_meta.exists() and json.loads(partial_meta.read_text()) != metadata:
                raise ValueError("Partial source changed")
            partial_meta.write_text(json.dumps(metadata), encoding="utf-8")
            for attempt in range(3):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > size:
                    raise ValueError("Partial file exceeds expected size")
                if offset == size:
                    break
                try:
                    with FTP() as ftp:
                        ftp.connect(HOST, timeout=90)
                        ftp.login(user, password)
                        ftp.voidcmd("TYPE I")
                        if ftp.size(row["remote_path"]) != size:
                            raise ValueError("Remote size changed since inventory")
                        with partial.open("ab") as stream:

                            def write(chunk):
                                with lock:
                                    if received[0] + len(chunk) > CAP:
                                        raise ValueError("Session download cap reached")
                                    received[0] += len(chunk)
                                if stream.tell() + len(chunk) > size:
                                    raise ValueError("Server exceeded declared file size")
                                stream.write(chunk)

                            ftp.retrbinary(
                                "RETR " + row["remote_path"],
                                write,
                                blocksize=256 * 1024,
                                rest=offset or None,
                            )
                    if partial.stat().st_size != size:
                        raise ValueError("Incomplete transfer")
                    break
                except (OSError, EOFError):
                    if attempt == 2:
                        raise
            if partial.stat().st_size != size:
                raise ValueError("Transfer incomplete after retries")
            with Image.open(partial) as im:
                im.verify()
            sha = digest(partial)
            partial.replace(target)
            record = {
                "source_url": url,
                "bytes": size,
                "sha256": sha,
                "acquired_at": datetime.now(UTC).isoformat(),
                "checksum_kind": "local_digest_and_server_size_not_provider_digest",
                "license": "non-commercial research; official portal terms supplied by user",
            }
            receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
            partial_meta.unlink(missing_ok=True)
        with Image.open(target) as im:
            im.verify()
        with Image.open(target) as im:
            width, height = im.size
        return {
            **row,
            "path": target.relative_to(ROOT).as_posix(),
            "sha256": record["sha256"],
            "width": width,
            "height": height,
        }

    def progress():
        status = {
            "status": "complete" if len(completed) == len(rows) else "in_progress_or_incomplete",
            "expected_images": len(rows),
            "verified_images": len(completed),
            "session_downloaded_bytes": received[0],
            "cap_bytes": CAP,
            "verified_bytes": sum(int(r["bytes"]) for r in completed),
            "failures": failures,
            "proposal_sha256": digest(proposal),
            "updated_at": datetime.now(UTC).isoformat(),
            "scope": "acquisition_only; no new training or test partition",
        }
        (out / "acquisition.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if completed:
            with (ROOT / "data/bracs_expansion_manifest.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=completed[0])
                writer.writeheader()
                writer.writerows(sorted(completed, key=lambda r: r["image_id"]))

    progress()
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(transfer, r): r for r in rows}
        for future in as_completed(pending):
            row = pending[future]
            try:
                completed.append(future.result())
                print(
                    f"{len(completed)}/{len(rows)} verified; {received[0] / 1e6:.1f} MB received",
                    flush=True,
                )
            except (OSError, EOFError, ValueError, Error, KeyError) as e:
                failures.append({"image": row["image_id"], "error": str(e)})
                print(f"Failed {row['image_id']}: {e}", flush=True)
            progress()
    if failures:
        raise RuntimeError(
            "Acquisition incomplete; see acquisition.json; rerun resumes verified partials"
        )
    print("Completed authorized 1 GB acquisition.", flush=True)


if __name__ == "__main__":
    main()
