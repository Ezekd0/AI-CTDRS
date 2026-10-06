import { useEffect, useState } from 'react';
import { api } from './api';
import { errorMessage } from './AccountPages';
import { runNetworkTest } from './networkTest';

const unavailable = 'Not available in this browser';
const value = (v, unit = '') => v == null ? unavailable : typeof v === 'number' ? `${v.toFixed(2)}${unit}` : v;
const date = v => new Date(v).toLocaleString();
const panel = 'rounded-xl border border-slate-800 bg-slate-900 p-5';
const button = 'rounded-lg bg-cyan-600 px-5 py-3 font-semibold disabled:opacity-50 hover:bg-cyan-500';
function fields(r) {
  return [
    ['Status', r.connectivity_status], ['Connection Type', r.connection_type],
    ['Internet Reachability', r.internet_status === 'reachable' ? 'AI-CTDRS backend reachable' : 'AI-CTDRS backend unreachable'],
    ['Latency', value(r.latency_ms, ' ms')], ['Packet Loss', unavailable],
    ['Download Speed', value(r.download_speed_mbps, ' Mbps')], ['Upload Speed', value(r.upload_speed_mbps, ' Mbps')],
    ['DNS/Reachability', r.dns_status], ['Effective Connection Type', r.metadata?.effective_type],
    ['Browser-reported RTT', value(r.metadata?.rtt_ms, ' ms')], ['Browser-reported Downlink', value(r.metadata?.downlink_mbps, ' Mbps')],
    ['Test Time', date(r.created_at)], ['Overall Network Condition', r.overall_status?.toUpperCase()],
  ];
}
function Diagnostic({ report }) {
  const [message, setMessage] = useState('');
  const text = ['AI-CTDRS NETWORK DIAGNOSTIC REPORT', ...fields(report).map(([label, v]) => `${label}: ${v ?? unavailable}`)].join('\n');
  return <section className={panel}><h2 className="font-semibold">AI-CTDRS NETWORK DIAGNOSTIC</h2><dl className="mt-4 grid gap-4 md:grid-cols-3">{fields(report).map(([label, v]) => <div key={label}><dt className="text-xs uppercase text-slate-500">{label}</dt><dd className="mt-1 text-sm">{v ?? unavailable}</dd></div>)}</dl><details className="mt-5"><summary className="cursor-pointer text-cyan-300">Provider report / Browser Network Info</summary><pre className="mt-3 whitespace-pre-wrap text-sm">{text}</pre><p className="mt-3 text-xs text-slate-400">HTTP probes: {report.metadata?.latency_samples_ms?.length ?? 0} succeeded; {report.metadata?.failed_http_requests ?? 0} failed. HTTP failures are not packet loss.</p><button className={`${button} mt-3`} onClick={async () => { try { await navigator.clipboard.writeText(text); setMessage('Report copied.'); } catch { setMessage('Copy unavailable. Select the report text above.'); } }}>Copy report</button><p role="status">{message}</p></details></section>;
}
function History({ reports, inspect, users = [] }) {
  return <section className={panel}><h2 className="font-semibold">Test history</h2>{!reports.length ? <p className="mt-3 text-slate-400">No saved network tests yet.</p> : <div className="overflow-x-auto"><table className="mt-3 w-full text-left text-sm"><thead><tr>{['Time', 'User', 'Latency', 'Download', 'Upload', 'Connectivity', 'Reachability', 'Condition', 'Report'].map(x => <th className="p-2" key={x}>{x}</th>)}</tr></thead><tbody>{reports.map(r => <tr key={r.id} className="border-t border-slate-800"><td className="p-2">{date(r.created_at)}</td><td className="p-2">{users.find(u => u.id === r.user_id)?.email || r.user_id}</td><td className="p-2">{value(r.latency_ms, ' ms')}</td><td className="p-2">{value(r.download_speed_mbps, ' Mbps')}</td><td className="p-2">{value(r.upload_speed_mbps, ' Mbps')}</td><td className="p-2">{r.connectivity_status}</td><td className="p-2">{r.internet_status}</td><td className="p-2 uppercase">{r.overall_status}</td><td className="p-2"><button className="text-cyan-300" onClick={() => inspect(r)}>Inspect</button></td></tr>)}</tbody></table></div>}</section>;
}
export function NetworkMonitorPage() {
  const [reports, setReports] = useState([]), [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false), [saving, setSaving] = useState(false), [error, setError] = useState('');
  const load = async () => {
    try { const { data } = await api.get('/network/reports', { params: { limit: 25 } }); setReports(data.items || []); setReport(r => r || data.items?.[0] || null); }
    catch (e) { setError(errorMessage(e, 'Unable to load saved reports. You can still run a test.')); }
  };
  useEffect(() => { load(); }, []);
  const run = async () => { setBusy(true); setError(''); try { setReport(await runNetworkTest()); } catch (e) { setError(errorMessage(e, 'Test interrupted. Try again.')); } finally { setBusy(false); } };
  const save = async () => { setSaving(true); setError(''); try { const { data } = await api.post('/network/reports', report); setReport(data); await load(); } catch (e) { setError(errorMessage(e, 'Save failed. Your result remains available; retry saving.')); } finally { setSaving(false); } };
  return <div className="space-y-5"><h1 className="text-2xl font-bold">Network Monitor</h1><p className="text-sm text-slate-400">Measure this browser’s connection to the AI-CTDRS backend.</p><button className={`${button} w-full py-5 text-lg`} disabled={busy || saving} onClick={run}>{busy ? 'Testing network…' : 'RUN NETWORK TEST'}</button>{busy && <p role="status">Running five HTTP probes and bounded download/upload transfers…</p>}{error && <p role="alert" className="text-red-300">{error}</p>}{report && <><Diagnostic report={report}/><button className={button} disabled={busy || saving || !!report.id} onClick={save}>{report.id ? 'Report saved' : saving ? 'Saving…' : 'Save report'}</button></>}<History reports={reports} inspect={setReport}/><section className={panel}><h2 className="font-semibold">How results are measured</h2><p className="mt-2 text-sm text-slate-400">Latency is the mean of successful authenticated HTTP requests. Speeds measure one 1 MB transfer in each direction, including HTTP/server overhead. Render cold starts can affect results. Reachability tests this backend only. Browsers cannot isolate DNS failures or measure packet loss. Browser Network Info is reported separately when supported.</p><p className="mt-3 text-sm text-slate-400">UNAVAILABLE: backend unreachable or a required measurement failed. POOR: latency ≥600 ms, download &lt;1 Mbps, or upload &lt;0.5 Mbps. DEGRADED: latency ≥250 ms, download &lt;3 Mbps, or upload &lt;1 Mbps. EXCELLENT: latency &lt;100 ms, download ≥25 Mbps and upload ≥5 Mbps. GOOD: other complete results. These are diagnostic thresholds, not cyberattack findings.</p></section></div>;
}
export function AdminNetworkReportsPanel() {
  const [reports, setReports] = useState([]), [users, setUsers] = useState([]), [selected, setSelected] = useState(''), [report, setReport] = useState(null), [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    api.get('/admin/users', { params: { limit: 200 } })
      .then(r => { if (active) setUsers(r.data.items || []); })
      .catch(e => { if (active) setError(errorMessage(e, 'Unable to load users.')); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    let active = true;
    setReport(null);
    setReports([]);
    setError('');
    api.get('/network/reports', { params: { limit: 100, ...(selected ? { user_id: selected } : {}) } })
      .then(r => { if (active) setReports(r.data.items || []); })
      .catch(e => { if (active) setError(errorMessage(e, 'Unable to load reports.')); });
    return () => { active = false; };
  }, [selected]);
  return <div className="space-y-5"><h1 className="text-2xl font-bold">Network Reports</h1>{error && <p role="alert">{error}</p>}<label className="block">Filter by user <select className="rounded border border-slate-700 bg-slate-900 p-2" value={selected} onChange={e => setSelected(e.target.value)}><option value="">All users</option>{users.map(u => <option key={u.id} value={u.id}>{u.email}</option>)}</select></label><History reports={reports} users={users} inspect={setReport}/>{report && <Diagnostic report={report}/>}</div>;
}
