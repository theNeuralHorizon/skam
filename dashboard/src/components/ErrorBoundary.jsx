import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, retryKey: 0 }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info.componentStack)
  }

  handleRetry = () => {
    this.setState(prev => ({
      hasError: false,
      error: null,
      retryKey: prev.retryKey + 1,
    }))
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--err)', marginBottom: 8 }}>
            Component Error
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 16 }}>
            {this.state.error?.message || 'Something went wrong rendering this panel.'}
          </div>
          <button className="sel" onClick={this.handleRetry}>
            Retry
          </button>
        </div>
      )
    }
    return <div key={this.state.retryKey}>{this.props.children}</div>
  }
}
