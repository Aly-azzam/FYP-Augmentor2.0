import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { evaluationHistory, scoreColor } from '@/services/mock/evaluation';
import { fetchCourses } from '@/services/api/courses';

function clipLabel(clipId: string) {
  const match = clipId.match(/clip-(\d+)$/);
  return match ? `Clip ${match[1]}` : clipId;
}

export default function HistoryPage() {
  const [search, setSearch] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [courseMap, setCourseMap] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;

    const loadCourses = async () => {
      try {
        const courses = await fetchCourses();
        if (!cancelled) {
          setCourseMap(Object.fromEntries(courses.map((course) => [course.id, course.title])));
        }
      } catch {
        if (!cancelled) {
          setCourseMap({});
        }
      }
    };

    void loadCourses();

    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    if (!q) return evaluationHistory;
    return evaluationHistory.filter(
      (ev) =>
        (courseMap[ev.courseId] ?? ev.courseId).toLowerCase().includes(q) ||
        clipLabel(ev.clipId).toLowerCase().includes(q),
    );
  }, [courseMap, search]);

  return (
    <div className="container section">
      <div className="page-header">
        <h1 className="page-title">Evaluation History</h1>
        <p className="page-subtitle">
          Review your past evaluations and track improvement
        </p>
      </div>

      {/* ── Search ────────────────────────────────────────────────────── */}
      <div style={{ position: 'relative', marginBottom: 'var(--space-xl)' }}>
        <Search
          size={18}
          style={{
            position: 'absolute',
            left: 12,
            top: '50%',
            transform: 'translateY(-50%)',
            color: 'var(--text-muted)',
          }}
        />
        <input
          className="input"
          placeholder="Search by course or clip..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ paddingLeft: 40 }}
        />
      </div>

      {/* ── Evaluation List ───────────────────────────────────────────── */}
      {filtered.length === 0 ? (
        <div className="empty-state">
          <FileText className="empty-state-icon" />
          <h3 className="empty-state-title">No evaluations found</h3>
          <p className="empty-state-description">
            {search
              ? 'Try a different search term.'
              : 'Complete an evaluation to see it here.'}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {filtered.map((ev, i) => {
            const isExpanded = expandedId === ev.id;
            const color = scoreColor(ev.score);

            return (
              <motion.div
                key={ev.id}
                className="card"
                style={{ overflow: 'hidden' }}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
              >
                {/* Header row */}
                <button
                  onClick={() => setExpandedId(isExpanded ? null : ev.id)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: 'var(--space-lg)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: 'inherit',
                    textAlign: 'left',
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div
                      className="text-small"
                      style={{ color: 'var(--text-muted)', marginBottom: 'var(--space-xs)' }}
                    >
                      {new Date(ev.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                    <div className="heading-4">
                      {courseMap[ev.courseId] ?? ev.courseId}
                    </div>
                    <div
                      className="text-small"
                      style={{ color: 'var(--text-secondary)', marginTop: 2 }}
                    >
                      {clipLabel(ev.clipId)}
                    </div>
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 'var(--space-md)',
                    }}
                  >
                    <span
                      className={`badge ${
                        ev.score >= 90
                          ? 'badge-green'
                          : ev.score >= 70
                            ? 'badge-blue'
                            : ev.score >= 50
                              ? 'badge-yellow'
                              : 'badge-red'
                      }`}
                    >
                      {ev.label}
                    </span>
                    <span
                      style={{
                        fontSize: '1.75rem',
                        fontWeight: 700,
                        color,
                        minWidth: 48,
                        textAlign: 'right',
                      }}
                    >
                      {ev.score}
                    </span>
                    {isExpanded ? (
                      <ChevronUp size={20} style={{ color: 'var(--text-muted)' }} />
                    ) : (
                      <ChevronDown size={20} style={{ color: 'var(--text-muted)' }} />
                    )}
                  </div>
                </button>

                {/* Expandable detail */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25 }}
                      style={{ overflow: 'hidden' }}
                    >
                      <div
                        style={{
                          padding: '0 var(--space-lg) var(--space-lg)',
                          borderTop: '1px solid var(--border-default)',
                          paddingTop: 'var(--space-lg)',
                        }}
                      >
                        {/* Metric grid */}
                        <div className="stats-grid" style={{ marginBottom: 'var(--space-lg)' }}>
                          {[
                            { label: 'Angle Deviation', value: ev.metrics.angleDeviation },
                            { label: 'Trajectory Deviation', value: ev.metrics.trajectoryDeviation },
                            { label: 'Velocity Difference', value: ev.metrics.velocityDifference },
                            { label: 'Tool Alignment', value: ev.metrics.toolAlignmentDeviation },
                          ].map((m) => (
                            <div key={m.label} className="stat-card">
                              <div className="stat-label">{m.label}</div>
                              <div
                                className="stat-value"
                                style={{ fontSize: '1.5rem' }}
                              >
                                {m.value.toFixed(1)}°
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* AI Explanation */}
                        <div
                          className="glass"
                          style={{
                            padding: 'var(--space-md)',
                            borderRadius: 'var(--radius-md)',
                          }}
                        >
                          <div
                            className="label"
                            style={{ marginBottom: 'var(--space-sm)' }}
                          >
                            AI Analysis
                          </div>
                          <p
                            className="text-body"
                            style={{ fontSize: '0.875rem', lineHeight: 1.7 }}
                          >
                            {ev.explanation}
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
