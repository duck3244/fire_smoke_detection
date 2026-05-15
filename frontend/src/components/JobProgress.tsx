import type { JobInfo } from '../types'

const statusLabel: Record<JobInfo['status'], string> = {
  queued: '대기 중',
  running: '처리 중',
  done: '완료',
  failed: '실패',
}

const statusColor: Record<JobInfo['status'], string> = {
  queued: 'bg-gray-200 text-gray-700',
  running: 'bg-blue-100 text-blue-800',
  done: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
}

export function JobProgress({ job }: { job: JobInfo }) {
  const pct = Math.round(job.progress * 100)
  return (
    <div className="rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm">
          Job <span className="font-mono text-xs text-gray-600">{job.job_id.slice(0, 12)}…</span>
        </div>
        <span className={`rounded px-2 py-0.5 text-xs font-medium ${statusColor[job.status]}`}>
          {statusLabel[job.status]}
        </span>
      </div>

      <div className="mt-3 h-2 w-full overflow-hidden rounded bg-gray-100">
        <div
          className={`h-full transition-all ${
            job.status === 'failed' ? 'bg-red-500' : 'bg-orange-500'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="mt-2 flex justify-between text-xs text-gray-600">
        <span>
          {job.processed_frames}/{job.total_frames} 프레임 ({pct}%)
        </span>
        <span>
          감지 프레임 {job.detection_frames} · 객체 {job.detections_total}
        </span>
      </div>

      {job.error && <div className="mt-2 text-sm text-red-600">⚠ {job.error}</div>}
    </div>
  )
}
