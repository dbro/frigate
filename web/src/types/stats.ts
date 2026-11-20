/** One local clock hour of the Secondlook disagreement summary. */
export type SecondlookHour = {
  hour_start: number;
  /** server-local hour start with offset, e.g. "2026-11-11T12:00+01:00" */
  local_hour: string;
  objects_analysed: number;
  false_positive: number;
  false_negative: number;
  misclass: number;
  /** FN split: released because of Secondlook, or not (answer too late). */
  fn_rescued: number;
  fn_not_rescued: number;
  /** FP split: kept from release by Secondlook, or released anyway. */
  fp_suppressed: number;
  fp_not_suppressed: number;
  /**
   * Per tracked object Secondlook was asked about, counted once in the hour it
   * finished (missing in responses from earlier builds). The five outcomes sum
   * to tracks_finished; agreements is a subset of results + no_hit.
   */
  tracks_finished?: number;
  agreements?: number;
  results?: number;
  no_hit?: number;
  timeouts?: number;
  dropped?: number;
  fail_open?: number;
};

/** A (primary, Secondlook) model pair in the hourly table; "" = unknown. */
export type SecondlookModelPair = {
  primary_model_id: string;
  sl_model_md5: string;
  first_hour: number;
  last_hour: number;
};

/** GET secondlook/disagreements/hourly: the last 168 hours, oldest first. */
export type SecondlookHourly = {
  timezone: string | null;
  hours: SecondlookHour[];
  models?: SecondlookModelPair[];
};

export interface FrigateStats {
  secondlook?: {
    enabled: boolean;
    available: boolean;
    samples: number;
    last_updated: number | null;
    inference_ms: number | null;
    processing_ms: number | null;
    queue_ms: number | null;
    device?: string;
    model_source?: "coco" | "frigate_plus" | null;
    model_refusal?: string | null;
    queue_dropped?: number;
    results_dropped?: number;
    /** Items now in Secondlook: frames waiting + the one in process (null without a real model). */
    queue_items?: number | null;
    /** Since the previous stats snapshot; null on the first one or without a real model. */
    interval?: {
      completed: number;
      dropped: number;
      /** most items at once, the one in process included */
      queue_peak: number;
    } | null;
    queue_max?: number;
    /** Corrected decisions since Frigate started (live counters, reset on restart). */
    accuracy?: {
      false_positive: number;
      false_negative: number;
      misclass: number;
    };
    storage?: {
      database_bytes: number;
      images_bytes: number;
      image_files: number;
      /** Last retention cleanup in this Frigate process; null until it ran. */
      retention: {
        last_run: number;
        deleted_rows: number;
        deleted_files: number;
        missing_files: number;
        deleted_orphans: number;
        deleted_summary_rows: number;
      } | null;
    };
  };
  cameras: { [camera_name: string]: CameraStats };
  cpu_usages: { [pid: string]: CpuStats };
  detectors: { [detectorKey: string]: DetectorStats };
  embeddings?: EmbeddingsStats;
  gpu_usages?: { [gpuKey: string]: GpuStats };
  npu_usages?: { [npuKey: string]: NpuStats };
  processes: { [processKey: string]: ExtraProcessStats };
  service: ServiceStats;
  camera_fps: number;
  process_fps: number;
  skipped_fps: number;
  detection_fps: number;
}

export type CameraStats = {
  /** Secondlook analyses completed per second (real inferences only). */
  secondlook_analysed_ps?: number;
  /** Secondlook analyses requested but not performed per second (queue full or crop failure). */
  secondlook_skipped_ps?: number;
  audio_dBFPS: number;
  audio_rms: number;
  camera_fps: number;
  capture_pid: number;
  detection_enabled: number;
  detection_fps: number;
  ffmpeg_pid: number;
  pid: number;
  process_fps: number;
  skipped_fps: number;
  connection_quality: "excellent" | "fair" | "poor" | "unusable";
  expected_fps: number;
  reconnects_last_hour: number;
  stalls_last_hour: number;
  ffmpeg_cpu?: string;
  capture_cpu?: string;
  detect_cpu?: string;
};

export type CpuStats = {
  cmdline: string;
  cpu: string;
  cpu_average: string;
  mem: string;
};

export type DetectorStats = {
  detection_start: number;
  inference_speed: number;
  pid: number;
  temperature?: number;
  cpu?: string;
  mem?: string;
};

export type EmbeddingsStats = {
  image_embedding_speed: number;
  face_embedding_speed: number;
  plate_recognition_speed: number;
  text_embedding_speed: number;
};

export type ExtraProcessStats = {
  pid: number;
  cpu?: string;
  mem?: string;
};

export type GpuVendor = "intel" | "amd" | "nvidia" | "rockchip" | "rpi";

export type GpuStats = {
  vendor?: GpuVendor;
  gpu: string;
  mem: string;
  enc?: string;
  dec?: string;
  compute?: string;
  pstate?: string;
  temp?: number;
  clients?: { [pid: string]: string };
};

export type NpuStats = {
  npu: number;
  mem: string;
  temp?: number;
};

export type GpuInfo = "vainfo" | "nvinfo";

export type ServiceStats = {
  last_updated: number;
  storage: { [path: string]: StorageStats };
  uptime: number;
  latest_version: string;
  version: string;
};

export type StorageStats = {
  free: number;
  total: number;
  used: number;
  mount_type: string;
  min_shm?: number;
  shm_frame_count?: number;
};

export type PotentialProblem = {
  text: string;
  color: string;
  relevantLink?: string;
};

export type Vainfo = {
  return_code: number;
  stdout: string;
  stderr: string;
};

export type Nvinfo = {
  [key: string]: {
    name: string;
    driver: string;
    cuda_compute: string;
    vbios: string;
  };
};

export type Ffprobe = {
  return_code: number;
  stderr: string;
  stdout: {
    programs: string[];
    streams: {
      avg_frame_rate: string;
      codec_long_name: string;
      height?: number;
      width?: number;
    }[];
  };
};

export type KeyframeSeverity =
  | "ok"
  | "warning"
  | "error"
  | "unknown"
  | "record_disabled";

export type KeyframeAnalysis = {
  severity: KeyframeSeverity;
  stream_index?: number;
  keyframe_count?: number;
  max_gap?: number | null;
  mean_gap?: number | null;
  min_gap?: number | null;
  duration_observed?: number | null;
  segment_time?: number;
  thresholds?: { warning: number; error: number };
};
