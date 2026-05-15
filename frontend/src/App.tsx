import { useState } from 'react'

import { ModelInfoCard } from './components/ModelInfoCard'
import { ImageDetect } from './pages/ImageDetect'
import { VideoDetect } from './pages/VideoDetect'

type Tab = 'image' | 'video'

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={[
        'rounded-md px-4 py-2 text-sm font-medium transition',
        active
          ? 'bg-orange-500 text-white shadow-sm'
          : 'bg-white text-gray-700 hover:bg-gray-50 border',
      ].join(' ')}
    >
      {children}
    </button>
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>('image')

  return (
    <div className="min-h-screen w-full bg-gray-50 text-gray-900">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <h1 className="text-lg font-semibold">🔥 Fire &amp; Smoke Detection</h1>
          <span className="text-xs text-gray-500">MVP · 단일 사용자</span>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-6 py-6">
        <ModelInfoCard />

        <div className="flex gap-2">
          <TabButton active={tab === 'image'} onClick={() => setTab('image')}>
            이미지
          </TabButton>
          <TabButton active={tab === 'video'} onClick={() => setTab('video')}>
            비디오
          </TabButton>
        </div>

        <section>{tab === 'image' ? <ImageDetect /> : <VideoDetect />}</section>
      </main>
    </div>
  )
}
