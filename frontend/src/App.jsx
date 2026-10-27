import { Fragment, useCallback, useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, XAxis, YAxis,
} from 'recharts'
import { api } from './api'
import { Card, ChartTip, Kpi, Severity, Tip, int, pct } from './ui'

const REFRESH_MS = 5000
const WINDOWS = [15, 30, 60, 180]
const clock = (s) => new Date(s * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function fillMinutes(rows, minutes) {
  const byMin = new Map(rows.map((r) => [r.minute, r]))
  const end = Math.floor(Date.now() / 60000) * 60
  const out = []
  for (let t = end - (minutes - 1) * 60; t <= end; t += 60) {
    const r = byMin.get(t)
    out.push({ t, flows: r?.flows ?? 0, alerts: r?.alerts ?? 0 })
  }
  return out
}

const axis = { tickLine: false, axisLine: false }

function HBars({ data, yKey, xKey, color = 'var(--flows)', fmt = int, height = 220 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" {...axis} tickFormatter={fmt} />
        <YAxis type="category" dataKey={yKey} width={118} {...axis} />
        <Tip content={<ChartTip fmtValue={fmt} />} />
        <Bar dataKey={xKey} name={xKey} fill={color} radius={[0, 4, 4, 0]} barSize={14} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function App() {
  const [minutes, setMinutes] = useState(30)
  const [stats, setStats] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [info, setInfo] = useState(null)
  const [online, setOnline] = useState(true)
  const [open, setOpen] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([api.stats(minutes), api.predictions(40, true)])
      setStats(s); setAlerts(a); setOnline(true)
    } catch { setOnline(false) }
  }, [minutes])

  useEffect(() => { refresh(); const id = setInterval(refresh, REFRESH_MS); return () => clearInterval(id) }, [refresh])
  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => {})
    api.info().then(setInfo).catch(() => {})
  }, [])

  const series = stats ? fillMinutes(stats.per_minute, Math.min(minutes, 60)) : []
  const t = metrics?.test
  const pr = metrics ? metrics.curves.pr.recall.map((r, i) => ({ recall: r, precision: metrics.curves.pr.precision[i] })) : []

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold tracking-tight">DriftGuard</h1>
        <span className="faint text-sm">Network intrusion detection · V1 static model</span>
        <div className="ml-auto flex flex-wrap items-center gap-3 text-xs">
          {info && <span className="card !py-1 !px-2 num">{info.version}</span>}
          <span className="inline-flex items-center gap-1.5" style={{ color: online ? 'var(--good)' : 'var(--crit)' }}>
            <span aria-hidden>{online ? '●' : '▲'}</span>{online ? 'live' : 'API unreachable'}
          </span>
        </div>
      </header>

      <div className="flex items-center gap-2 text-sm" role="group" aria-label="Time window">
        <span className="faint">Window</span>
        {WINDOWS.map((w) => (
          <button key={w} onClick={() => setMinutes(w)}
            className="px-2.5 py-1 rounded-md border num"
            style={{ borderColor: 'var(--border)', background: w === minutes ? 'var(--text)' : 'var(--surface)', color: w === minutes ? 'var(--surface)' : 'var(--text-2)' }}>
            {w < 60 ? `${w}m` : `${w / 60}h`}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Kpi label={`Flows scored (${minutes}m)`} value={int(stats?.window_flows)} hint={`${int(stats?.total_flows)} all time`} />
        <Kpi label={`Alerts (${minutes}m)`} value={int(stats?.window_alerts)} tone="alerts" hint={`${int(stats?.total_alerts)} all time`} />
        <Kpi label="Alert rate" value={pct(stats?.window_alert_rate, 2)} hint="share of flows flagged" />
        <Kpi label="False-alarm budget" value={pct(metrics?.fpr_budget, 1)} hint="benign flows allowed to alert" />
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Flows per minute" sub="Traffic volume scored by the model">
          <ResponsiveContainer width="100%" height={190}>
            <LineChart data={series} margin={{ left: -8, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="t" {...axis} tickFormatter={clock} minTickGap={36} />
              <YAxis {...axis} tickFormatter={int} />
              <Tip content={<ChartTip fmtLabel={clock} />} />
              <Line type="monotone" dataKey="flows" name="Flows" stroke="var(--flows)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Alerts per minute" sub="Flows the model flagged as potentially malicious">
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={series} margin={{ left: -8, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="t" {...axis} tickFormatter={clock} minTickGap={36} />
              <YAxis {...axis} allowDecimals={false} />
              <Tip cursor={{ fill: 'var(--grid)' }} content={<ChartTip fmtLabel={clock} />} />
              <Bar dataKey="alerts" name="Alerts" fill="var(--alerts)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <Card title="Recent alerts" sub="Newest first. Click a row for the flow's features" className="lg:col-span-2">
          <div className="overflow-x-auto max-h-[360px] overflow-y-auto">
            <table className="dg">
              <thead><tr><th>Time</th><th>Severity</th><th>Probability</th><th>Dst port</th><th>Flow</th><th>Ground truth</th></tr></thead>
              <tbody>
                {alerts.length === 0 && <tr><td colSpan={6} className="faint">No alerts yet. Start <code>scripts/simulate.py</code> to stream traffic.</td></tr>}
                {alerts.map((a) => (
                  <Fragment key={a.id}>
                    <tr onClick={() => setOpen(open === a.id ? null : a.id)} className="cursor-pointer hover:bg-[var(--grid)]">
                      <td className="num">{new Date(a.ts * 1000).toLocaleTimeString()}</td>
                      <td><Severity level={a.severity} /></td>
                      <td className="num">{pct(a.probability, 0)}</td>
                      <td className="num">{a.dst_port}</td>
                      <td className="faint">{a.flow_id ?? a.id}</td>
                      <td className="faint">{a.true_label == null ? '–' : a.true_label ? 'attack' : 'benign (false alarm)'}</td>
                    </tr>
                    {open === a.id && (
                      <tr><td colSpan={6} className="faint num text-xs" style={{ background: 'var(--bg)' }}>
                        {Object.entries(a.features).map(([k, v]) => `${k}=${v}`).join('  ·  ')}
                      </td></tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Top alerted ports" sub={`Last ${minutes} minutes`}>
          {stats?.top_alert_ports?.length
            ? <HBars data={stats.top_alert_ports.map((p) => ({ ...p, port: `port ${p.port}` }))} yKey="port" xKey="alerts" color="var(--alerts)" height={Math.max(160, stats.top_alert_ports.length * 30)} />
            : <p className="faint text-sm">No alerts in this window.</p>}
        </Card>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Model confidence" sub="Flows per calibrated attack-probability bucket (log scale: each gridline is 10x more flows)">
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={(stats?.probability_histogram ?? []).map((b) => ({ ...b, log: Math.log10(b.count + 1) }))} margin={{ left: -8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="bin" {...axis} />
              <YAxis {...axis} domain={[0, (m) => Math.max(1, Math.ceil(m))]} allowDecimals={false} tickFormatter={(v) => int(Math.round(10 ** v - 1))} />
              <Tip cursor={{ fill: 'var(--grid)' }} content={<ChartTip fmtValue={(v) => int(Math.round(10 ** v - 1))} />} />
              <Bar dataKey="log" name="Flows" fill="var(--flows)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Why rarity hurts precision" sub="Same detector, different attack rates (exact calculation from the test TPR and FPR)">
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={[...(metrics?.precision_vs_prevalence ?? [])].reverse().map((r) => ({ ...r, label: pct(r.prevalence, r.prevalence < 0.01 ? 1 : 0) }))} margin={{ left: -8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="label" {...axis} />
              <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} />
              <Tip cursor={{ fill: 'var(--grid)' }} content={<ChartTip fmtLabel={(l) => `attack rate ${l}`} fmtValue={(v) => pct(v, 1)} />} />
              <Bar dataKey="precision" name="Precision" fill="var(--flows)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <h2 className="text-base font-semibold pt-2">Model evaluation <span className="faint font-normal text-sm">held-out test block, natural attack rate {pct(metrics?.data?.attack_rate, 1)}</span></h2>
      {metrics?.caveats?.map((c) => (
        <p key={c} className="card text-sm" style={{ borderColor: 'var(--warn)', color: 'var(--text-2)' }}><span style={{ color: 'var(--warn)' }}>▲ </span>{c}</p>
      ))}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <Kpi label="PR-AUC" value={t ? t.pr_auc.toFixed(3) : '–'} />
        <Kpi label="Recall (attacks caught)" value={pct(t?.recall)} />
        <Kpi label="Precision (alerts that are real)" value={pct(t?.precision)} />
        <Kpi label="False-positive rate" value={pct(t?.fpr, 2)} />
        <Kpi label="F1" value={t ? t.f1.toFixed(3) : '–'} />
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card title="Precision–recall curve" sub="Dot marks the deployed operating point">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={pr} margin={{ left: -8, right: 12, bottom: 18 }}>
              <CartesianGrid />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} {...axis} tickFormatter={(v) => pct(v, 0)} label={{ value: 'Recall', position: 'insideBottom', offset: -2, fill: 'var(--text-3)', fontSize: 11 }} />
              <YAxis type="number" domain={[0, 1]} {...axis} tickFormatter={(v) => pct(v, 0)} />
              <Tip content={<ChartTip fmtLabel={(l) => `recall ${pct(l, 0)}`} fmtValue={(v) => pct(v, 1)} />} />
              <Line dataKey="precision" name="Precision" stroke="var(--flows)" strokeWidth={2} dot={false} isAnimationActive={false} />
              {t && <ReferenceDot x={t.recall} y={t.precision} r={5} fill="var(--alerts)" stroke="var(--surface)" strokeWidth={2} />}
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Recall by attack type" sub="Share of each attack family detected at the deployed threshold">
          <HBars data={metrics?.per_attack_type ?? []} yKey="attack_type" xKey="recall" fmt={(v) => pct(v, 0)} height={240} />
        </Card>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <Card title="Imbalance strategies compared" sub={`Same model and test set; only the imbalance handling differs. Selected on validation data: ${metrics?.selected_strategy ?? '–'}`} className="lg:col-span-2">
          <div className="overflow-x-auto">
            <table className="dg">
              <thead><tr><th>Strategy</th><th>PR-AUC</th><th>Recall</th><th>Precision</th><th>FPR</th><th>Recall @1% FPR</th></tr></thead>
              <tbody>
                {metrics?.comparison.map((r) => (
                  <tr key={r.strategy} style={r.strategy === metrics.selected_strategy ? { fontWeight: 600 } : undefined}>
                    <td>{r.strategy}{r.strategy === metrics.selected_strategy && <span className="faint"> ✓ deployed</span>}</td>
                    <td className="num">{r.pr_auc.toFixed(3)}</td><td className="num">{pct(r.recall)}</td>
                    <td className="num">{pct(r.precision)}</td><td className="num">{pct(r.fpr, 2)}</td>
                    <td className="num">{pct(r['recall_at_1%_fpr'])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="What drives detection" sub="Permutation importance (drop in average precision)">
          <HBars data={(metrics?.feature_importance ?? []).slice(0, 8)} yKey="feature" xKey="importance" fmt={(v) => v.toFixed(2)} height={250} />
        </Card>
      </div>

      <footer className="faint text-xs pt-2 pb-6">
        Data source: {metrics?.data?.source ?? '–'} · trained {metrics?.trained_at ?? '–'} · refreshes every {REFRESH_MS / 1000}s
      </footer>
    </div>
  )
}
