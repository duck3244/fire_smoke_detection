export interface Detection {
  cls: string
  confidence: number
  bbox: [number, number, number, number]
}

export interface ImageDetectionResult {
  image_id: string
  annotated_url: string
  detections: Detection[]
  inference_ms: number
  width: number
  height: number
}

export interface JobCreated {
  job_id: string
  status: 'queued'
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed'

export interface JobInfo {
  job_id: string
  status: JobStatus
  progress: number
  processed_frames: number
  total_frames: number
  detection_frames: number
  detections_total: number
  result_url: string | null
  error: string | null
}

export interface ModelInfo {
  weights_path: string
  classes: string[]
  num_classes: number
  device: string
  map50: number | null
  map50_95: number | null
  precision: number | null
  recall: number | null
}
