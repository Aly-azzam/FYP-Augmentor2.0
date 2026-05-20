import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ChevronDown, ChevronUp, FileText, Loader2 } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface HistoryEntry {
  attempt_id: string;
  chapter_id: string;
  course_id: string;
  course_title: string;
  chapter_title: string;
  status: string;
  score: number | null;
  created_at: string;
}

function scoreColor(score: number | null) {
  if (score === null) return 'var(--text-muted)';
  if (score >= 90) return '#22c55e';
  if (score >= 70) return '#3b82f6';
  if (score >= 50) return '#f59e0b';
  return '#ef4444';
}

function scoreLabel(score: number | null) {
  if (score === null) return 'Pending';
  if (score >= 90) return 'Excellent';
  if (score >= 70) return 'Good';
  if (score >= 50) return 'Fair';
  return 'Needs Work';
}

function scoreBadgeClass(score: number | null) {
  if (score === null) return '';
  if (score >= 90) return 'badge-green';
  if (score >= 70) return 'badge-blue';
  if (score >= 50) return 'badge-yellow';
  return 'badge-red';
}

export default function HistoryPage() {
  const { user, token } = useAuth();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    if (!user || !token) return;
    setLoading(true);
    fetch(`${API}/api/history?user_id=${user.id}&limit=100`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load history');
        return r.json();
      })
      .then(setEntries)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [user, token]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    if (!q) return entries;
    return entries.filter(
      (ev) =>
        ev.course_title.toLowerCase().includes(q) ||
        ev.chapter_title.toLowerCase().includes(q),
    );
  }, [entries, search]);

  return (
    <div className="container section">
      <div className="page-header">
        <h1 className="page-title">Evaluation History</h1>
        <p className="page-subtitle">Review your past evaluations and track improvement</p>
      </div>

      <div style={{ position: 'relative', marginBottom: 'var(--space-xl)' }}>
        <Search size={18} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
        <input className="input" placeholder="Search by course or clip..." value={search} onChange={(e) => setSearch(e.target.value)} style={{ paddingLeft: 40 }} />
      </div>

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 60 }}>
          <Loader2 size={32} style={{ color: 'var(--text-muted)', animation: 'spin 1s linear infinite' }} />
        </div>
      )}

      {error && (
        <div className="card" style={{ padding: 'var(--space-lg)', color: '#ef4444', textAlign: 'center' }}>{error}</div>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="empty-state">
          <FileText className="empty-state-icon" />
          <h3 className="empty-state-title">No evaluations found</h3>
          <p className="empty-state-description">
            {search ? 'Try a different search term.' : 'Complete an evaluation in Compare Studio to see it here.'}
          </p>
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {filtered.map((ev, i) => {
            const isExpanded = expandedId === ev.attempt_id;
            const color = scoreColor(ev.score);
            return (
              <motion.div key={ev.attempt_id} className="card" style={{ overflow: 'hidden' }} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                <button
                  onClick={() => setExpandedId(isExpanded ? null : ev.attempt_id)}
                  style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 'var(--space-lg)', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', textAlign: 'left' }}
                >
                  <div style={{ flex: 1 }}>
                    <div className="text-small" style={{ color: 'var(--text-muted)', marginBottom: 'var(--space-xs)' }}>
                      {new Date(ev.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div className="heading-4">{ev.course_title}</div>
                    <div className="text-small" style={{ color: 'var(--text-secondary)', marginTop: 2 }}>{ev.chapter_title}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
                    <span className={`badge ${scoreBadgeClass(ev.score)}`}>{scoreLabel(ev.score)}</span>
                    <span style={{ fontSize: '1.75rem', fontWeight: 700, color, minWidth: 48, textAlign: 'right' }}>
                      {ev.score !== null ? ev.score : '—'}
                    </span>
                    {isExpanded ? <ChevronUp size={20} style={{ color: 'var(--text-muted)' }} /> : <ChevronDown size={20} style={{ color: 'var(--text-muted)' }} />}
                  </div>
                </button>

                <AnimatePresence>
                  {isExpanded && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25 }} style={{ overflow: 'hidden' }}>
                      <div style={{ padding: '0 var(--space-lg) var(--space-lg)', borderTop: '1px solid var(--border-default)', paddingTop: 'var(--space-lg)' }}>
                        <div className="stats-grid" style={{ marginBottom: 'var(--space-lg)' }}>
                          {[
                            { label: 'Status', value: ev.status },
                            { label: 'Score', value: ev.score !== null ? `${ev.score}%` : '—' },
                            { label: 'Course', value: ev.course_title },
                            { label: 'Chapter', value: ev.chapter_title },
                          ].map((m) => (
                            <div key={m.label} className="stat-card">
                              <div className="stat-label">{m.label}</div>
                              <div className="stat-value" style={{ fontSize: '1rem', wordBreak: 'break-word' }}>{m.value}</div>
                            </div>
                          ))}
                        </div>
                        <div className="glass" style={{ padding: 'var(--space-md)', borderRadius: 'var(--radius-md)' }}>
                          <div className="label" style={{ marginBottom: 'var(--space-sm)' }}>AI Analysis</div>
                          <p className="text-body" style={{ fontSize: '0.875rem', lineHeight: 1.7, color: 'var(--text-muted)' }}>
                            Full AI feedback and error breakdown are available in Compare Studio after completing an evaluation.
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
