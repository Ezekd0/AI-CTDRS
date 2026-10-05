import { useEffect, useState } from 'react';
import { api } from './api';

export function errorMessage(error, fallback) {
  const detail = error.response?.data?.errors || error.response?.data?.detail;
  return typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(x => `${x.loc?.slice(1).join('.') || 'Input'}: ${x.msg}`).join('; ') : fallback;
}
const inputClass = 'w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2.5 outline-none focus:border-cyan-500';
const buttonClass = 'rounded-lg bg-cyan-600 px-4 py-2.5 font-semibold hover:bg-cyan-500 disabled:opacity-50';

export function SignupPage({ navigate }) {
  const [values, setValues] = useState({ full_name: '', email: '', password: '', confirm: '' });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [success, setSuccess] = useState(false);
  async function submit(event) {
    event.preventDefault();
    setError('');
    if (!values.full_name.trim()) return setError('Enter your full name.');
    if (values.password.length < 12 || values.password.length > 128) return setError('Password must contain 12–128 characters.');
    if (values.password !== values.confirm) return setError('Passwords do not match.');
    setBusy(true);
    try {
      await api.post('/auth/register', { full_name: values.full_name.trim(), email: values.email.trim(), password: values.password });
      setValues({ full_name: '', email: '', password: '', confirm: '' });
      setSuccess(true);
    } catch (err) { setError(errorMessage(err, 'Unable to create your account. Please try again.')); }
    finally { setBusy(false); }
  }
  return <div className="flex min-h-screen items-center justify-center bg-slate-950 px-5 text-slate-100"><section className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-7 shadow-2xl"><div className="text-xs font-bold tracking-[.22em] text-cyan-400">AI-CTDRS</div><h1 className="mt-3 text-2xl font-bold">Create account</h1>{success ? <div role="status" className="mt-6 text-emerald-300">Registration successful. Sign in with your new account.</div> : <form onSubmit={submit} className="mt-6 space-y-4">{[['full_name', 'Full name', 'text', 'name'], ['email', 'Email', 'email', 'email'], ['password', 'Password', 'password', 'new-password'], ['confirm', 'Confirm password', 'password', 'new-password']].map(([key, label, type, autoComplete]) => <label key={key} className="block text-sm"><span className="mb-2 block text-slate-400">{label}</span><input required type={type} autoComplete={autoComplete} maxLength={key === 'full_name' ? 200 : key === 'email' ? 320 : 128} value={values[key]} onChange={e => setValues({ ...values, [key]: e.target.value })} className={inputClass}/></label>)}<p className="text-xs text-slate-400">Use a password with 12–128 characters.</p>{error && <div role="alert" className="text-sm text-red-300">{error}</div>}<button disabled={busy} className={`${buttonClass} w-full`}>{busy ? 'Creating account…' : 'Sign up'}</button></form>}<a href="/login" onClick={e => { e.preventDefault(); navigate('/login'); }} className="mt-5 block text-sm text-cyan-300">{success ? 'Return to Login' : 'Already have an account? Sign in'}</a></section></div>;
}

export function AdminOverview({ navigate, openDetection }) {
  const [data, setData] = useState(null), [error, setError] = useState('');
  useEffect(() => { let active = true; api.get('/admin/overview').then(r => { if (active) setData(r.data); }).catch(e => { if (active) setError(errorMessage(e, 'Unable to load admin overview.')); }); return () => { active = false; }; }, []);
  return <div className="space-y-5"><h1 className="text-2xl font-bold">Admin dashboard</h1>{error && <p role="alert" className="text-red-300">{error}</p>}{!data && !error && <p role="status">Loading overview…</p>}{data && <><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{[['Total users', data.total_users], ['Active users', data.active_users], ['Registrations (last 7 days)', data.registrations_last_7_days], ['Total detections', data.total_detections]].map(([label, value]) => <section key={label} className="rounded-xl border border-slate-800 bg-slate-900 p-5"><p className="text-sm text-slate-400">{label}</p><p className="mt-2 text-2xl font-semibold">{value}</p></section>)}</div><section className="rounded-xl border border-slate-800 bg-slate-900 p-5"><h2 className="font-semibold">System status</h2><p className="mt-3 text-sm text-emerald-300">API: {data.system_status.api} · Database: {data.system_status.database} · Response mode: {data.system_status.response_mode}</p></section><section className="rounded-xl border border-slate-800 bg-slate-900 p-5"><h2 className="font-semibold">Recent registrations</h2>{data.recent_registrations.map(u => <div key={u.id} className="mt-3 flex flex-wrap justify-between gap-2 text-sm"><span>{u.full_name || 'Name not provided'} · {u.email}</span><span className="text-slate-400">{new Date(u.created_at).toLocaleString()}</span></div>)}</section><section className="rounded-xl border border-slate-800 bg-slate-900 p-5"><h2 className="font-semibold">Recent detections</h2>{data.recent_detections.length ? data.recent_detections.map(d => <button key={d.id} onClick={() => openDetection(d.id)} className="mt-3 block text-sm text-cyan-300">{d.prediction} · {new Date(d.created_at).toLocaleString()}</button>) : <p className="mt-3 text-sm text-slate-400">No detections yet.</p>}</section><a href="/admin/users" onClick={e => { e.preventDefault(); navigate('/admin/users'); }} className="inline-block text-cyan-300">Manage users →</a></>}</div>;
}

export function AdminUsers({ currentUser }) {
  const [filters, setFilters] = useState({ search: '', role: '', is_active: '' });
  const [offset, setOffset] = useState(0), [data, setData] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true), [error, setError] = useState(''), [pending, setPending] = useState(''), [notice, setNotice] = useState(''), [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true; setLoading(true); setError('');
    const timer = setTimeout(() => { const params = { offset, limit: 25 }; Object.entries(filters).forEach(([k, v]) => { if (v !== '') params[k] = v; }); api.get('/admin/users', { params }).then(r => { if (active) setData(r.data); }).catch(e => { if (active) setError(errorMessage(e, 'Unable to load users.')); }).finally(() => { if (active) setLoading(false); }); }, 200);
    return () => { active = false; clearTimeout(timer); };
  }, [filters, offset, revision]);
  async function toggle(user) {
    setPending(user.id); setError(''); setNotice('');
    try { await api.patch(`/admin/users/${user.id}/status`, { is_active: !user.is_active }); setNotice(`${user.email} ${user.is_active ? 'disabled' : 'activated'}.`); setRevision(x => x + 1); }
    catch (e) { setError(errorMessage(e, 'Unable to update this user.')); }
    finally { setPending(''); }
  }
  const change = (key, value) => { setOffset(0); setFilters(f => ({ ...f, [key]: value })); };
  return <div className="space-y-5"><h1 className="text-2xl font-bold">User management</h1><div className="grid gap-3 md:grid-cols-3"><input aria-label="Search users" placeholder="Search name or email" className={inputClass} value={filters.search} onChange={e => change('search', e.target.value)}/><select aria-label="Filter role" className={inputClass} value={filters.role} onChange={e => change('role', e.target.value)}><option value="">All roles</option>{['administrator', 'analyst', 'viewer'].map(role => <option key={role}>{role}</option>)}</select><select aria-label="Filter status" className={inputClass} value={filters.is_active} onChange={e => change('is_active', e.target.value)}><option value="">All statuses</option><option value="true">Active</option><option value="false">Disabled</option></select></div>{error && <p role="alert" className="text-red-300">{error}</p>}{notice && <p role="status" className="text-emerald-300">{notice}</p>}{loading ? <p role="status">Loading users…</p> : !error && <><div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900"><table className="w-full text-left text-sm"><thead><tr>{['Name', 'Email', 'Role', 'Status', 'Registered', 'Action'].map(label => <th className="p-4" key={label}>{label}</th>)}</tr></thead><tbody>{data.items.map(user => <tr key={user.id} className="border-t border-slate-800"><td className="p-4">{user.full_name || 'Name not provided'}</td><td className="p-4">{user.email}</td><td className="p-4">{user.role}</td><td className="p-4">{user.is_active ? 'Active' : 'Disabled'}</td><td className="p-4">{new Date(user.created_at).toLocaleString()}</td><td className="p-4">{user.role === 'administrator' || user.id === currentUser.id ? 'Protected' : <button disabled={!!pending} onClick={() => toggle(user)} className={buttonClass}>{pending === user.id ? 'Saving…' : user.is_active ? 'Disable' : 'Activate'}</button>}</td></tr>)}</tbody></table>{!data.items.length && <p className="p-5 text-slate-400">No matching users.</p>}</div><div className="flex items-center gap-4 text-sm"><button disabled={!offset} onClick={() => setOffset(x => Math.max(0, x - 25))}>Previous</button><span>{data.total ? offset + 1 : 0}–{Math.min(offset + 25, data.total)} of {data.total}</span><button disabled={offset + 25 >= data.total} onClick={() => setOffset(x => x + 25)}>Next</button></div></>}</div>;
}
