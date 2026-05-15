import type {
  ImageDetectionResult,
  JobCreated,
  JobInfo,
  ModelInfo,
} from '../types'

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init)
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const data = await res.json()
      if (data?.detail) detail = `${detail}: ${data.detail}`
    } catch {
      // ignore json parse error
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export async function getModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>('/api/model/info')
}

export async function detectImage(file: File): Promise<ImageDetectionResult> {
  const form = new FormData()
  form.append('file', file)
  return request<ImageDetectionResult>('/api/detect/image', {
    method: 'POST',
    body: form,
  })
}

export async function startVideoJob(file: File): Promise<JobCreated> {
  const form = new FormData()
  form.append('file', file)
  return request<JobCreated>('/api/detect/video', {
    method: 'POST',
    body: form,
  })
}

export async function getJob(jobId: string): Promise<JobInfo> {
  return request<JobInfo>(`/api/jobs/${jobId}`)
}

export function jobResultUrl(jobId: string): string {
  return `/api/jobs/${jobId}/result`
}
