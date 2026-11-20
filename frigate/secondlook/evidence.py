"""Coherent analysed-frame evidence. Coordinates are full-frame XYXY pixels."""

import base64
import hashlib

import cv2


def source_evidence(frame, box, region, frame_time, primary_score):
    image = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok or jpeg.nbytes > 4 * 1024 * 1024:
        return None
    return {
        "version": 1,
        "frame_time": frame_time,
        "width": image.shape[1],
        "height": image.shape[0],
        "primary_box": list(box),
        "crop_region": list(region),
        "primary_score": primary_score,
        "jpeg": base64.b64encode(jpeg).decode("ascii"),
    }


def full_frame_box(box, region, width, height):
    if box is None:
        return None
    y1, x1, y2, x2 = box
    rx1, ry1, rx2, ry2 = region
    return [
        max(0, min(width, rx1 + x1 * (rx2 - rx1))),
        max(0, min(height, ry1 + y1 * (ry2 - ry1))),
        max(0, min(width, rx1 + x2 * (rx2 - rx1))),
        max(0, min(height, ry1 + y2 * (ry2 - ry1))),
    ]


def finish_evidence(
    source, sl_box, label, score, model_id, *, camera=None, primary_label=None
):
    if source is None:
        return None
    evidence = dict(source)
    evidence.update(
        sl_box=full_frame_box(
            sl_box, source["crop_region"], source["width"], source["height"]
        ),
        secondary_label=label if score is not None else None,
        secondary_score=score,
        model_id=model_id,
    )
    if camera is not None:
        evidence.update(camera=camera, label=primary_label)
    return evidence


def jpeg_bytes(evidence):
    return base64.b64decode(evidence["jpeg"], validate=True)


def metadata(evidence):
    return {k: v for k, v in evidence.items() if k != "jpeg"}


def group_key(
    jpeg, camera, kind, label, secondary, model_id, primary_model_id, verdict=None
):
    return group_digest(
        hashlib.sha256(jpeg).hexdigest(),
        camera,
        kind,
        label,
        secondary,
        model_id,
        primary_model_id,
        verdict,
    )


def group_digest(
    digest, camera, kind, label, secondary, model_id, primary_model_id, verdict
):
    import json

    return (
        "v2:"
        + hashlib.sha256(
            json.dumps(
                [
                    digest,
                    camera,
                    kind,
                    label,
                    secondary,
                    model_id,
                    primary_model_id,
                    verdict,
                ],
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )


from functools import lru_cache


@lru_cache(maxsize=256)
def existing_image_digest(path, mtime, size):
    digest = hashlib.sha256()
    with open(path, "rb") as image:
        for chunk in iter(lambda: image.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
