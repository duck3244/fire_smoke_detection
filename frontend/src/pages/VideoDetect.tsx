import { useEffect, useRef, useState } from 'react'

import { getJob, jobResultUrl, startVideoJob } from '../api/client'
import { DropZone } from '../components/DropZone'
import { JobProgress } from '../components/JobProgress'
import type { JobInfo } from '../types'

const POLL_MS = 800

export function VideoDetect() {
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState<JobInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  // 폴링
  useEffect(() => {
    if (!job || job.status === 'done' || job.status === 'failed') {
      if (pollRef.current) window.clearTimeout(pollRef.current)
      return
    }
    pollRef.current = window.setTimeout(async () => {
      try {
        const next = await getJob(job.job_id)
        setJob(next)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      }
    }, POLL_MS)
    return () => {
      if (pollRef.current) window.clearTimeout(pollRef.current)
    }
  }, [job])

  async function handle(file: File) {
    setBusy(true)
    setError(null)
    setJob(null)
    setFilename(file.name)
    try {
      const created = await startVideoJob(file)
      // 첫 상태 즉시 가져와 폴링 시작
      const first = await getJob(created.job_id)
      setJob(first)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const done = job?.status === 'done'

  return (
    <div className="space-y-4">
      <DropZone
        accept=".mp4,.mov,.avi,.mkv,.webm"
        label={busy ? '업로드 중…' : '비디오 업로드'}
        disabled={busy || (job !== null && job.status === 'running')}
        onFile={handle}
      />

      {filename && (
        <div className="text-xs text-gray-500">
          업로드: <span className="font-mono">{filename}</span>
        </div>
      )}

      {error && (
        <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {job && <JobProgress job={job} />}

      {done && job.result_url && (
        <div className="rounded-lg border bg-white p-2">
          <video
            src={job.result_url}
            controls
            className="mx-auto max-h-[600px] w-auto rounded"
          />
          <div className="mt-2 text-right">
            <a
              href={jobResultUrl(job.job_id)}
              download
              className="text-sm text-orange-600 hover:underline"
            >
              결과 다운로드 (mp4)
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
