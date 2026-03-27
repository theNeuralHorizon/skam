import { Activity, Wifi, WifiOff } from 'lucide-react'

interface HeaderProps {
  isConnected: boolean
  healthyCount: number
  totalServices: number
}

export function Header({ isConnected, healthyCount, totalServices }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-950">
      <div className="flex items-center gap-3">
        <Activity className="w-7 h-7 text-orange-500" />
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">SKAM</h1>
          <p className="text-xs text-gray-500">Chaos Engineering &amp; Self-Healing Platform</p>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-400">Services:</span>
          <span className={healthyCount === totalServices ? 'text-green-400' : 'text-yellow-400'}>
            {healthyCount}/{totalServices} healthy
          </span>
        </div>

        <div className="flex items-center gap-2 text-sm">
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4 text-green-400" />
              <span className="text-green-400">Live</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-red-400" />
              <span className="text-red-400">Disconnected</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
