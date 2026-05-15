import { useState } from 'react'

import { detectImage } from '../api/client'
import { DetectionTable } from '../components/DetectionTable'
import { DropZone } from '../components/DropZone'
import type { ImageDetectionResult } from '../types'

export function ImageDetect() {
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ImageDetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)

  async function handle(file: File) {
    setBusy(true)
    setError(null)
    setResult(null)
    setFilename(file.name)
    try {
      const r = await detectImage(file)
      setResult(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <DropZone
        accept=".jpg,.jpeg,.png,.bmp,.webp"
        label={busy ? '추론 중…' : '이미지 업로드'}
        disabled={busy}
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

      {result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>
              {result.width}×{result.height} · 감지 {result.detections.length}개
            </span>
            <span className="tabular-nums">추론 {result.inference_ms.toFixed(1)}ms</span>
          </div>

          <div className="rounded-lg border bg-white p-2">
            <img
              src={result.annotated_url}
              alt="annotated"
              className="mx-auto max-h-[600px] w-auto rounded"
            />
          </div>

          <div className="rounded-lg border bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-800">감지 결과</h3>
            <DetectionTable detections={result.detections} />
          </div>
        </div>
      )}
    </div>
  )
}
