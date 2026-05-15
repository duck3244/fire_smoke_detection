import { useEffect, useState } from 'react'

import { getModelInfo } from '../api/client'
import type { ModelInfo } from '../types'

function pct(x: number | null) {
  return x == null ? '—' : `${(x * 100).toFixed(1)}%`
}

export function ModelInfoCard() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError(String(e)))
  }, [])

  if (error) {
    return <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700">{error}</div>
  }
  if (!info) {
    return <div className="rounded border bg-gray-50 p-3 text-sm text-gray-500">모델 정보 로딩…</div>
  }

  return (
    <div className="rounded border bg-white p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">모델 정보</h3>
        <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700">
          device: <span className="font-mono">{info.device}</span>
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <div className="text-gray-500">클래스</div>
          <div className="font-medium">{info.classes.join(', ')}</div>
        </div>
        <div>
          <div className="text-gray-500">mAP@0.5</div>
          <div className="font-medium tabular-nums">{pct(info.map50)}</div>
        </div>
        <div>
          <div className="text-gray-500">mAP@0.5:0.95</div>
          <div className="font-medium tabular-nums">{pct(info.map50_95)}</div>
        </div>
        <div>
          <div className="text-gray-500">Precision / Recall</div>
          <div className="font-medium tabular-nums">
            {pct(info.precision)} · {pct(info.recall)}
          </div>
        </div>
      </div>
      <div className="mt-2 truncate font-mono text-xs text-gray-500" title={info.weights_path}>
        {info.weights_path}
      </div>
    </div>
  )
}
