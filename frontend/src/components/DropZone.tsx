import { useCallback, useRef, useState } from 'react'

interface Props {
  accept: string
  label: string
  disabled?: boolean
  onFile: (file: File) => void
}

export function DropZone({ accept, label, disabled, onFile }: Props) {
  const [over, setOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handle = useCallback(
    (file: File | null) => {
      if (!file) return
      onFile(file)
    },
    [onFile],
  )

  return (
    <div
      onDragOver={(e) => {
        if (disabled) return
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        if (disabled) return
        e.preventDefault()
        setOver(false)
        handle(e.dataTransfer.files?.[0] ?? null)
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={[
        'flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 text-center transition cursor-pointer select-none',
        disabled
          ? 'border-gray-300 bg-gray-100 text-gray-400 cursor-not-allowed'
          : over
            ? 'border-orange-500 bg-orange-50 text-orange-700'
            : 'border-gray-300 hover:border-orange-400 hover:bg-orange-50 text-gray-600',
      ].join(' ')}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0] ?? null)}
      />
      <div className="text-lg font-medium">{label}</div>
      <div className="mt-1 text-sm">클릭하거나 파일을 끌어다 놓으세요</div>
      <div className="mt-2 text-xs">허용: {accept}</div>
    </div>
  )
}
