import type { Detection } from '../types'

const classColor: Record<string, string> = {
  Fire: 'bg-red-100 text-red-800 border-red-300',
  smoke: 'bg-gray-200 text-gray-700 border-gray-300',
}

export function DetectionTable({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return <div className="text-sm text-gray-500">감지된 객체가 없습니다.</div>
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50 text-left">
          <tr>
            <th className="px-3 py-2 font-medium text-gray-600">#</th>
            <th className="px-3 py-2 font-medium text-gray-600">Class</th>
            <th className="px-3 py-2 font-medium text-gray-600">Confidence</th>
            <th className="px-3 py-2 font-medium text-gray-600">BBox (x1,y1,x2,y2)</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((d, i) => (
            <tr key={i} className="border-t">
              <td className="px-3 py-2 text-gray-500">{i + 1}</td>
              <td className="px-3 py-2">
                <span
                  className={`inline-block rounded border px-2 py-0.5 text-xs font-medium ${
                    classColor[d.cls] ?? 'bg-blue-100 text-blue-800 border-blue-300'
                  }`}
                >
                  {d.cls}
                </span>
              </td>
              <td className="px-3 py-2 tabular-nums">{(d.confidence * 100).toFixed(1)}%</td>
              <td className="px-3 py-2 font-mono text-xs text-gray-600">
                {d.bbox.map((v) => v.toFixed(0)).join(', ')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
