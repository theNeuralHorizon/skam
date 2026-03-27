import { useCallback, useEffect, useRef, useState } from 'react'

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 15000,
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setInterval>>()

  const poll = useCallback(async () => {
    try {
      const result = await fetcher()
      setData(result)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Fetch failed')
    } finally {
      setLoading(false)
    }
  }, [fetcher])

  useEffect(() => {
    poll()
    timer.current = setInterval(poll, intervalMs)
    return () => clearInterval(timer.current)
  }, [poll, intervalMs])

  return { data, loading, error, refresh: poll }
}
