import logging
import os
from enum import Enum

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from frigate.const import MODEL_CACHE_DIR
from frigate.util.yolo_decode_config import YoloDecodeConfig

from .base import FrigateBaseModel

logger = logging.getLogger(__name__)

__all__ = [
    "CameraFaceRecognitionConfig",
    "CameraLicensePlateRecognitionConfig",
    "CameraAudioTranscriptionConfig",
    "CameraSecondlookConfig",
    "FaceRecognitionConfig",
    "SemanticSearchConfig",
    "CameraSemanticSearchConfig",
    "LicensePlateRecognitionConfig",
    "SecondlookConfig",
    "SecondlookModelConfig",
    "SecondlookPairedFiltersConfig",
]


class SemanticSearchModelEnum(str, Enum):
    jinav1 = "jinav1"
    jinav2 = "jinav2"


class EnrichmentsDeviceEnum(str, Enum):
    GPU = "GPU"
    CPU = "CPU"


class ModelSizeEnum(str, Enum):
    small = "small"
    large = "large"


class TriggerType(str, Enum):
    THUMBNAIL = "thumbnail"
    DESCRIPTION = "description"


class TriggerAction(str, Enum):
    NOTIFICATION = "notification"
    SUB_LABEL = "sub_label"
    ATTRIBUTE = "attribute"


class ObjectClassificationType(str, Enum):
    sub_label = "sub_label"
    attribute = "attribute"


class AudioTranscriptionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable audio transcription",
        description="Enable or disable automatic audio transcription for all cameras; can be overridden per-camera.",
    )
    language: str = Field(
        default="en",
        title="Transcription language",
        description="Language code used for transcription/translation (for example 'en' for English). See https://whisper-api.com/docs/languages/ for supported language codes.",
    )
    device: EnrichmentsDeviceEnum = Field(
        default=EnrichmentsDeviceEnum.CPU,
        title="Transcription device",
        description="Device key (CPU/GPU) to run the transcription model on. Only NVIDIA CUDA GPUs are currently supported for transcription.",
    )
    model_size: ModelSizeEnum = Field(
        default=ModelSizeEnum.small,
        title="Model size",
        description="Model size to use for offline audio event transcription.",
    )
    live_enabled: bool | None = Field(
        default=False,
        title="Live transcription",
        description="Enable streaming live transcription for audio as it is received.",
    )


class BirdClassificationConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Bird classification",
        description="Enable or disable bird classification.",
    )
    threshold: float = Field(
        default=0.9,
        title="Minimum score",
        description="Minimum classification score required to accept a bird classification.",
        gt=0.0,
        le=1.0,
    )


class CustomClassificationStateCameraConfig(FrigateBaseModel):
    crop: list[float, float, float, float] = Field(
        title="Classification crop",
        description="Crop coordinates to use for running classification on this camera.",
    )


class CustomClassificationStateConfig(FrigateBaseModel):
    cameras: dict[str, CustomClassificationStateCameraConfig] = Field(
        title="Classification cameras",
        description="Per-camera crop and settings for running state classification.",
    )
    motion: bool = Field(
        default=False,
        title="Run on motion",
        description="If true, run classification when motion is detected within the specified crop.",
    )
    interval: int | None = Field(
        default=None,
        title="Classification interval",
        description="Interval (seconds) between periodic classification runs for state classification.",
        gt=0,
    )


class CustomClassificationObjectConfig(FrigateBaseModel):
    objects: list[str] = Field(
        default_factory=list,
        title="Classify objects",
        description="List of object types to run object classification on.",
    )
    classification_type: ObjectClassificationType = Field(
        default=ObjectClassificationType.sub_label,
        title="Classification type",
        description="Classification type applied: 'sub_label' (adds sub_label) or other supported types.",
    )


class CustomClassificationConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=True,
        title="Enable model",
        description="Enable or disable the custom classification model.",
    )
    name: str | None = Field(
        default=None,
        title="Model name",
        description="Identifier for the custom classification model to use.",
    )
    threshold: float = Field(
        default=0.8,
        title="Score threshold",
        description="Score threshold used to change the classification state.",
    )
    save_attempts: int | None = Field(
        default=None,
        title="Save attempts",
        description="How many classification attempts to save for recent classifications UI.",
        ge=0,
    )
    object_config: CustomClassificationObjectConfig | None = Field(default=None)
    state_config: CustomClassificationStateConfig | None = Field(default=None)


class ClassificationConfig(FrigateBaseModel):
    bird: BirdClassificationConfig = Field(
        default_factory=BirdClassificationConfig,
        title="Bird classification config",
        description="Settings specific to bird classification models.",
    )
    custom: dict[str, CustomClassificationConfig] = Field(
        default={},
        title="Custom Classification Models",
        description="Configuration for custom classification models used for objects or state detection.",
    )


class SemanticSearchConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable semantic search",
        description="Enable or disable the semantic search feature.",
    )
    reindex: bool | None = Field(
        default=False,
        title="Reindex on startup",
        description="Trigger a full reindex of historical tracked objects into the embeddings database.",
    )
    model: SemanticSearchModelEnum | str | None = Field(
        default=SemanticSearchModelEnum.jinav1,
        title="Semantic search model or GenAI provider name",
        description="The embeddings model to use for semantic search (for example 'jinav1'), or the name of a GenAI provider with the embeddings role.",
    )

    @field_validator("model", mode="before")
    @classmethod
    def coerce_model_enum(cls, v):
        if isinstance(v, str):
            try:
                return SemanticSearchModelEnum(v)
            except ValueError:
                return v
        return v

    model_size: ModelSizeEnum = Field(
        default=ModelSizeEnum.small,
        title="Model size",
        description="Select model size; 'small' runs on CPU and 'large' typically requires GPU.",
    )
    device: str | None = Field(
        default=None,
        title="Device",
        description="This is an override, to target a specific device. See https://onnxruntime.ai/docs/execution-providers/ for more information",
    )


class TriggerConfig(FrigateBaseModel):
    friendly_name: str | None = Field(
        None,
        title="Friendly name",
        description="Optional friendly name displayed in the UI for this trigger.",
    )
    enabled: bool = Field(
        default=True,
        title="Enable this trigger",
        description="Enable or disable this semantic search trigger.",
    )
    type: TriggerType = Field(
        default=TriggerType.DESCRIPTION,
        title="Trigger type",
        description="Type of trigger: 'thumbnail' (match against image) or 'description' (match against text).",
    )
    data: str = Field(
        title="Trigger content",
        description="Text phrase or thumbnail ID to match against tracked objects.",
    )
    threshold: float = Field(
        title="Trigger threshold",
        description="Minimum similarity score (0-1) required to activate this trigger.",
        default=0.8,
        gt=0.0,
        le=1.0,
    )
    actions: list[TriggerAction] = Field(
        default=[],
        title="Trigger actions",
        description="List of actions to execute when trigger matches (notification, sub_label, attribute).",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class CameraSemanticSearchConfig(FrigateBaseModel):
    triggers: dict[str, TriggerConfig] = Field(
        default={},
        title="Triggers",
        description="Actions and matching criteria for camera-specific semantic search triggers.",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class FaceRecognitionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable face recognition",
        description="Enable or disable face recognition for all cameras; can be overridden per-camera.",
    )
    model_size: ModelSizeEnum = Field(
        default=ModelSizeEnum.small,
        title="Model size",
        description="Model size to use for face embeddings (small/large); larger may require GPU.",
    )
    unknown_score: float = Field(
        title="Unknown score threshold",
        description="Distance threshold below which a face is considered a potential match (higher = stricter).",
        default=0.8,
        gt=0.0,
        le=1.0,
    )
    detection_threshold: float = Field(
        default=0.7,
        title="Detection threshold",
        description="Minimum detection confidence required to consider a face detection valid.",
        gt=0.0,
        le=1.0,
    )
    recognition_threshold: float = Field(
        default=0.9,
        title="Recognition threshold",
        description="Face embedding distance threshold to consider two faces a match.",
        gt=0.0,
        le=1.0,
    )
    min_area: int = Field(
        default=750,
        title="Minimum face area",
        description="Minimum area (pixels) of a detected face box required to attempt recognition.",
    )
    min_faces: int = Field(
        default=1,
        gt=0,
        le=6,
        title="Minimum faces",
        description="Minimum number of face recognitions required before applying a recognized sub-label to a person.",
    )
    save_attempts: int = Field(
        default=200,
        ge=0,
        title="Save attempts",
        description="Number of face recognition attempts to retain for recent recognition UI.",
    )
    blur_confidence_filter: bool = Field(
        default=True,
        title="Blur confidence filter",
        description="Adjust confidence scores based on image blur to reduce false positives for poor quality faces.",
    )
    device: str | None = Field(
        default=None,
        title="Device",
        description="This is an override, to target a specific device. See https://onnxruntime.ai/docs/execution-providers/ for more information",
    )


class CameraFaceRecognitionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable face recognition",
        description="Enable or disable face recognition.",
    )
    min_area: int = Field(
        default=750,
        title="Minimum face area",
        description="Minimum area (pixels) of a detected face box required to attempt recognition.",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ReplaceRule(FrigateBaseModel):
    pattern: str = Field(..., title="Regex pattern")
    replacement: str = Field(..., title="Replacement string")


class LicensePlateRecognitionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable LPR",
        description="Enable or disable license plate recognition for all cameras; can be overridden per-camera.",
    )
    model_size: ModelSizeEnum = Field(
        default=ModelSizeEnum.small,
        title="Model size",
        description="Model size used for text detection/recognition. Most users should use 'small'.",
    )
    detection_threshold: float = Field(
        default=0.7,
        title="Detection threshold",
        description="Detection confidence threshold to begin running OCR on a suspected plate.",
        gt=0.0,
        le=1.0,
    )
    min_area: int = Field(
        default=1000,
        title="Minimum plate area",
        description="Minimum plate area (pixels) required to attempt recognition.",
    )
    recognition_threshold: float = Field(
        default=0.9,
        title="Recognition threshold",
        description="Confidence threshold required for recognized plate text to be attached as a sub-label.",
        gt=0.0,
        le=1.0,
    )
    min_plate_length: int = Field(
        default=4,
        title="Min plate length",
        description="Minimum number of characters a recognized plate must contain to be considered valid.",
    )
    format: str | None = Field(
        default=None,
        title="Plate format regex",
        description="Optional regex to validate recognized plate strings against an expected format.",
    )
    match_distance: int = Field(
        default=1,
        title="Match distance",
        description="Number of character mismatches allowed when comparing detected plates to known plates.",
        ge=0,
    )
    known_plates: dict[str, list[str]] | None = Field(
        default={},
        title="Known plates",
        description="List of plates or regexes to specially track or alert on.",
    )
    enhancement: int = Field(
        default=0,
        title="Enhancement level",
        description="Enhancement level (0-10) to apply to plate crops prior to OCR; higher values may not always improve results, levels above 5 may only work with night time plates and should be used with caution.",
        ge=0,
        le=10,
    )
    debug_save_plates: bool = Field(
        default=False,
        title="Save debug plates",
        description="Save plate crop images for debugging LPR performance.",
    )
    device: str | None = Field(
        default=None,
        title="Device",
        description="This is an override, to target a specific device. See https://onnxruntime.ai/docs/execution-providers/ for more information",
    )
    replace_rules: list[ReplaceRule] = Field(
        default_factory=list,
        title="Replacement rules",
        description="Regex replacement rules used to normalize detected plate strings before matching.",
    )


class CameraLicensePlateRecognitionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable LPR",
        description="Enable or disable LPR on this camera.",
    )
    expire_time: int = Field(
        default=3,
        title="Expire seconds",
        description="Time in seconds after which an unseen plate is expired from the tracker (for dedicated LPR cameras only).",
        gt=0,
    )
    min_area: int = Field(
        default=1000,
        title="Minimum plate area",
        description="Minimum plate area (pixels) required to attempt recognition.",
    )
    enhancement: int = Field(
        default=0,
        title="Enhancement level",
        description="Enhancement level (0-10) to apply to plate crops prior to OCR; higher values may not always improve results, levels above 5 may only work with night time plates and should be used with caution.",
        ge=0,
        le=10,
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class CameraAudioTranscriptionConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable transcription",
        description="Enable or disable manually triggered audio event transcription.",
    )
    enabled_in_config: bool | None = Field(
        default=None, title="Original transcription state"
    )
    live_enabled: bool | None = Field(
        default=False,
        title="Live transcription",
        description="Enable streaming live transcription for audio as it is received.",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class SecondlookModelConfig(FrigateBaseModel, YoloDecodeConfig):
    """Secondlook model files plus YOLO post-processing settings (min_score,
    nms_threshold, max_detections up to 300, nms_class_groups)."""

    path: str = Field(
        default="/config/model_cache/secondlook/yolov9-c-640.onnx",
        title="Secondlook model path",
        description="Path to the yolov9-c-640 ONNX model used for Secondlook inference.",
    )
    labelmap_path: str = Field(
        default="/config/model_cache/secondlook/labelmap.txt",
        title="Secondlook labelmap path",
        description="COCO-80 labelmap for the Secondlook YOLO model (one label per line or 'id label').",
    )
    input_size: int = Field(
        default=640,
        title="Input size",
        description="Square input size expected by the Secondlook model. Ignored when plus_metadata_path is set.",
        gt=0,
    )
    plus_metadata_path: str | None = Field(
        default=None,
        title="Frigate+ model info",
        description="Use a Frigate+ YOLOv9 ONNX model from a local file: path to its Frigate+ model info JSON (for example /config/model_cache/<model_id>.json next to path /config/model_cache/<model_id>). Labels, input size, layout, pixel format and dtype are read from it; labelmap_path and input_size are ignored. An unsuitable model is refused at startup and Secondlook fails open. path: plus://<model_id> fills both in; nothing is downloaded.",
    )

    @model_validator(mode="after")
    def resolve_plus_path(self) -> "SecondlookModelConfig":
        """path: plus://<id> -> /config/model_cache/<id> (+ <id>.json). No download."""
        if not self.path.startswith("plus://"):
            return self
        model_id = self.path[len("plus://") :]
        if not model_id or "/" in model_id or model_id in (".", ".."):
            raise ValueError(f"invalid Secondlook plus:// model id {model_id!r}")
        path = os.path.join(MODEL_CACHE_DIR, model_id)
        # plain object.__setattr__: no re-validation loop on assignment
        object.__setattr__(self, "path", path)
        if self.plus_metadata_path is None:
            object.__setattr__(self, "plus_metadata_path", f"{path}.json")
        return self


class SecondlookPairedFiltersConfig(FrigateBaseModel):
    min_score: float | None = Field(
        default=0.3,
        title="Paired min_score",
        description="When Secondlook is enabled, override objects.filters.*.min_score for SL object labels. Stock default is 0.5. Set null to leave stock min_score alone.",
        gt=0.0,
        le=1.0,
    )
    min_initialized: int | None = Field(
        default=None,
        title="Paired min_initialized",
        description="Optional override for detect.min_initialized when Secondlook is enabled. Stock is often fps/2 (min 2). Lowering to 1 helps weak hits register for SL.",
        ge=1,
    )


class SecondlookSaveAndReviewDatabaseConfig(FrigateBaseModel):
    path: str = Field(
        default="/config/secondlook_examples.db",
        title="Database path",
        description="Separate SQLite file. The existing filename is retained so saved records stay available.",
    )


class SecondlookSaveAndReviewImagesConfig(FrigateBaseModel):
    path: str = Field(
        default="/media/frigate/clips/secondlook_examples",
        title="Images path",
        description="Full-frame JPEG directory. Existing images are not moved.",
    )


class SecondlookSaveAndReviewReviewConfig(FrigateBaseModel):
    group_gap_seconds: float = Field(
        default=60.0,
        ge=0.0,
        title="Grouping time gap",
        description="Corrections Review groups saved decisions with the same camera, kind, primary class and Secondlook class when consecutive frame times are no more than this many seconds apart. Display only: every saved row, image and mark is kept. 0 groups only decisions with an identical frame time.",
    )
    hide_edge_small: bool = Field(
        default=True,
        title="Hide small edge boxes by default",
        description="Default state of the Review toggle that hides decisions whose primary box touches the frame edge and is small. Filter only; data is still saved. Records without saved frame size and box stay visible.",
    )
    edge_margin_px: int = Field(
        default=4,
        ge=0,
        title="Edge margin (pixels)",
        description="A primary box counts as touching the frame edge when any side is within this many source-frame pixels of the edge.",
    )
    small_dim_fraction: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        title="Small box dimension fraction",
        description="A primary box counts as small when its width is below this fraction of the frame width, or its height is below this fraction of the frame height (0.05 = 1/20; about 64 px wide or 36 px tall on 1280x720).",
    )

    @model_validator(mode="before")
    @classmethod
    def _ignore_removed_small_area_fraction(cls, data):
        # FrigateBaseModel forbids unexpected keys. Accept the previous
        # small_area_fraction so existing YAML still starts, then ignore it.
        if isinstance(data, dict) and "small_area_fraction" in data:
            data = dict(data)
            data.pop("small_area_fraction")
            logger.warning(
                "secondlook.save_and_review.review.small_area_fraction is no longer "
                "used and was ignored. Small boxes are now those whose width is "
                "below small_dim_fraction of the frame width or whose height is "
                "below that fraction of the frame height (default 0.05)."
            )
        return data


class SecondlookSaveAndReviewRetentionConfig(FrigateBaseModel):
    unreviewed_days: float | None = Field(
        default=30.0,
        ge=0.0,
        title="Keep unreviewed decisions (days)",
        description="Delete unreviewed saved decisions this many days after capture. 0 or null keeps them.",
    )
    ignored_days: float | None = Field(
        default=7.0,
        ge=0.0,
        title="Keep ignored decisions (days)",
        description="Delete decisions marked Ignore this many days after they were last marked. 0 or null keeps them.",
    )
    max_size_gb: float | None = Field(
        default=None,
        ge=0.0,
        title="Size cap (GiB)",
        description="If the database plus images exceed this many GiB, delete the oldest unreviewed, then the oldest ignored decisions until under it. 0 or null disables. Primary right / Secondlook right decisions are never deleted automatically.",
    )
    summary_days: float | None = Field(
        default=90.0,
        ge=0.0,
        title="Keep hourly disagreement summary (days)",
        description="Delete hourly Secondlook disagreement summary rows older than this. 0 or null keeps them.",
    )
    analyses_days: float | None = Field(
        default=90.0,
        ge=0.0,
        title="Keep per-object analyses (days)",
        description="Delete per-object Secondlook analysis rows (one per tracked object Secondlook was asked about, no images) whose object ended more than this many days ago. 0 or null keeps them.",
    )


class SecondlookSaveAndReviewConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=True,
        title="Enable save and review",
        description="Save data and enable the Corrections Review experience. Independent of live Secondlook score adjustment. Restart after changing this setting.",
    )
    database: SecondlookSaveAndReviewDatabaseConfig = Field(
        default_factory=SecondlookSaveAndReviewDatabaseConfig,
    )
    images: SecondlookSaveAndReviewImagesConfig = Field(
        default_factory=SecondlookSaveAndReviewImagesConfig,
    )
    review: SecondlookSaveAndReviewReviewConfig = Field(
        default_factory=SecondlookSaveAndReviewReviewConfig,
        title="Corrections Review display",
    )
    retention: SecondlookSaveAndReviewRetentionConfig = Field(
        default_factory=SecondlookSaveAndReviewRetentionConfig,
        title="Retention",
        description="Daily cleanup (and once at startup) of unreviewed and ignored decisions. Accepted decisions are never deleted automatically.",
    )


class SecondlookConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable Secondlook",
        description="Enable Secondlook live score adjustment globally; cameras must also enable secondlook.",
    )
    device: EnrichmentsDeviceEnum = Field(
        default=EnrichmentsDeviceEnum.GPU,
        title="Inference device",
        description="OpenVINO device for Secondlook (GPU = iGPU, CPU fallback).",
    )
    model: SecondlookModelConfig = Field(
        default_factory=SecondlookModelConfig,
        title="Secondlook model",
        description="ONNX model path and labelmap for Secondlook.",
    )
    hold_seconds: float = Field(
        default=1.0,
        title="Hold seconds",
        description="Max time to keep a track false_positive while waiting for the first Secondlook result (fail-open after).",
        gt=0.0,
    )
    refresh_every_n: int = Field(
        default=10,
        title="Refresh every N hits",
        description="After N real hit frames, re-run Secondlook on the max-primary-score crop and update the factor.",
        ge=1,
    )
    context_scale: float = Field(
        default=1.5,
        title="Context scale",
        description="Multiplier passed to Frigate calculate_region when cropping for Secondlook (region floored at model.input_size for more scene context, not upscale).",
        gt=1.0,
    )
    factor_min: float = Field(
        default=0.3,
        title="Factor clamp min",
        description="Minimum scale factor applied to the primary scores in the score buffer before detection.",
        gt=0.0,
    )
    factor_max: float = Field(
        default=2.0,
        title="Factor clamp max",
        description="Maximum scale factor applied to the primary scores in the score buffer before detection.",
        gt=0.0,
    )
    queue_max: int = Field(
        default=8,
        title="In-flight queue max",
        description="Max queued Secondlook inference jobs; overflow fails open to unscaled primary.",
        ge=1,
    )
    objects: list[str] = Field(
        default_factory=lambda: ["person", "car", "dog", "cat"],
        title="Secondlook objects",
        description="Object labels eligible for Secondlook live QA.",
    )
    paired_filters: SecondlookPairedFiltersConfig = Field(
        default_factory=SecondlookPairedFiltersConfig,
        title="Paired admission filters",
        description="Admission overrides applied only while Secondlook is enabled (starting points to tune).",
    )
    score_threshold: float = Field(
        default=0.25,
        title="SL detection score threshold",
        description="Minimum Secondlook same-class detection score to count as a hit on the crop.",
        gt=0.0,
        le=1.0,
    )
    no_hit_factor: float | None = Field(
        default=0.0,
        title="No-hit factor",
        description="Factor applied when Secondlook finds no same-class detection on the crop. Default 0.0 = SL decisively overrules the primary (hard suppress until an overlapping same-class hit). null leaves the factor unset (= 1.0, unscaled primary).",
        ge=0.0,
    )
    min_box_iou: float | None = Field(
        default=0.3,
        title="Min SL/track box IoU",
        description="Optional: only count SL same-class boxes whose IoU with the track's own box (inside the crop) is at least this. Default 0.3. null counts any same-class box in the crop.",
        ge=0.0,
        le=1.0,
    )

    save_and_review: SecondlookSaveAndReviewConfig = Field(
        default_factory=SecondlookSaveAndReviewConfig,
        title="Save and review",
        description="Save data for human review and future fine-tuning, and enable the Corrections tab. Does not train models or control live score adjustment.",
    )
    gap_factor_low: float = Field(
        default=0.8,
        title="FP gap factor",
        description="Deposit as False positive when factor is below this and primary would have cleared threshold.",
        gt=0.0,
    )
    gap_factor_high: float = Field(
        default=1.2,
        title="FN gap factor",
        description="Deposit as False negative when factor is above this and primary alone stayed under threshold.",
        gt=0.0,
    )


class CameraSecondlookConfig(FrigateBaseModel):
    # Copied from global config; not a per-camera YAML option.
    _save_and_review_enabled: bool = PrivateAttr(default=False)
    _gap_factor_low: float = PrivateAttr(default=0.8)
    _gap_factor_high: float = PrivateAttr(default=1.2)

    enabled: bool = Field(
        default=False,
        title="Enable Secondlook",
        description="Enable Secondlook live score adjustment for this camera (requires global secondlook.enabled).",
    )
    objects: list[str] | None = Field(
        default=None,
        title="Secondlook objects",
        description="Per-camera object labels for Secondlook; null inherits the global list.",
    )
    # Runtime knobs copied from global during config merge so the false_positive check can read them.
    hold_seconds: float = Field(default=1.0, title="Hold seconds", gt=0.0)
    factor_min: float = Field(default=0.3, title="Factor clamp min", gt=0.0)
    factor_max: float = Field(default=2.0, title="Factor clamp max", gt=0.0)

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
