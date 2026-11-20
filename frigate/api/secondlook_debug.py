"""An explicit analysed-frame snapshot, never asynchronous boxes on live video."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from frigate.api.auth import get_allowed_cameras_for_filter

router = APIRouter(tags=["Secondlook"])


@router.get("/{camera_name}/secondlook")
def analysed_frame(
    request: Request,
    camera_name: str,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    config = request.app.frigate_config
    if camera_name not in allowed_cameras:
        return JSONResponse({"message": "Camera not allowed"}, status_code=403)
    camera = config.cameras.get(camera_name)
    if not camera or not config.secondlook.enabled or not camera.secondlook.enabled:
        return {"state": "disabled", "frame": None}
    state = request.app.detected_frames_processor.camera_states.get(camera_name)
    frame = getattr(state, "latest_secondlook", None)
    return {"state": "ready" if frame else "no_result", "frame": frame}
