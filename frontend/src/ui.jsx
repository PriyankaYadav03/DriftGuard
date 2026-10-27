import { Tooltip as RTooltip } from 'recharts'

export const pct = (v, d = 1) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`)
export const int = (v) => (v == null ? '–' : Number(v).toLocaleString())

export function Card({ title, sub, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {title && <h2 className="text-sm font-semibold">{title}</h2>}
      {sub && <p className="faint text-xs mt-0.5 mb-3">{sub}</p>}
      {!sub && title && <div className="mb-3" />}
      {children}
    </section>
  )
}

export function Kpi({ label, value, hint, tone }) {
  return (
    <div className="card">
      <div className="faint text-xs">{label}</div>
      <div className="num text-2xl font-semibold mt-1" style={tone ? { color: `var(--${tone})` } : undefined}>{value}</div>
      {hint && <div className="faint text-xs mt-1">{hint}</div>}
    </div>
  )
}

const SEV = {
  high: { icon: '▲', color: 'var(--crit)' },
  medium: { icon: '◆', color: 'var(--warn)' },
  low: { icon: '●', color: 'var(--text-2)' },
}
export function Severity({ level }) {
  const s = SEV[level]
  if (!s) return <span className="faint">–</span>
  return <span className="inline-flex items-center gap-1.5 text-xs font-medium" style={{ color: s.color }}><span aria-hidden>{s.icon}</span>{level}</span>
}

export function ChartTip({ active, payload, label, fmtLabel, fmtValue = int }) {
  if (!active || !payload?.length) return null
  return (
    <div className="card !p-2 text-xs shadow-md" style={{ borderRadius: 8 }}>
      <div className="faint mb-1">{fmtLabel ? fmtLabel(label) : label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <span style={{ background: p.color || p.stroke || p.fill, width: 8, height: 8, borderRadius: 2 }} />
          <span className="muted">{p.name}</span>
          <span className="num ml-auto pl-3 font-medium">{fmtValue(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

export const Tip = (props) => <RTooltip cursor={{ stroke: 'var(--border)' }} {...props} />
