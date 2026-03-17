import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';
import { TrendingUp, Target, Flame, Award } from 'lucide-react';
import { userProfile, progressData } from '@/services/mock/user';
import { courses } from '@/services/mock/courses';

const ACCENT = '#2563EB';

const tooltipStyle = {
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-md)',
  color: 'var(--text-primary)',
};

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function ProgressPage() {
  const radarData = useMemo(() => {
    const n = progressData.length;
    const avg = (key: 'metric1' | 'metric2' | 'metric3' | 'metric4') =>
      progressData.reduce((sum, d) => sum + d[key], 0) / n;

    return [
      { metric: 'Angle', value: Math.round(Math.max(0, 100 - avg('metric1') * 5)) },
      { metric: 'Trajectory', value: Math.round(Math.max(0, 100 - avg('metric2') * 5)) },
      { metric: 'Velocity', value: Math.round(Math.max(0, 100 - avg('metric3') * 5)) },
      { metric: 'Alignment', value: Math.round(Math.max(0, 100 - avg('metric4') * 5)) },
    ];
  }, []);

  const activeCourses = courses.filter((c) => c.progress > 0);

  const stats = [
    { icon: TrendingUp, label: 'Total Evaluations', value: userProfile.totalEvaluations },
    { icon: Target, label: 'Average Score', value: userProfile.averageScore },
    { icon: Flame, label: 'Current Streak', value: `${userProfile.streakDays} days` },
    { icon: Award, label: 'Level', value: userProfile.level },
  ];

  return (
    <div className="container section">
      <div className="page-header">
        <h1 className="page-title">My Learning</h1>
        <p className="page-subtitle">
          Track your progress and see how you're improving over time
        </p>
      </div>

      {/* ── Stat Cards ──────────────────────────────────────────────────── */}
      <div className="stats-grid">
        {stats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              className="card stat-card"
              {...fadeUp}
              transition={{ delay: i * 0.05 }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-sm)',
                  marginBottom: 'var(--space-sm)',
                }}
              >
                <Icon size={20} style={{ color: ACCENT }} />
                <span className="stat-label">{stat.label}</span>
              </div>
              <div className="stat-value">{stat.value}</div>
            </motion.div>
          );
        })}
      </div>

      {/* ── Area Chart – Score Over Time ─────────────────────────────── */}
      <motion.div
        className="card"
        style={{ padding: 'var(--space-lg)', marginTop: 'var(--space-xl)' }}
        {...fadeUp}
        transition={{ delay: 0.1 }}
      >
        <h3 className="heading-4" style={{ marginBottom: 'var(--space-lg)' }}>
          Score Over Time
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={progressData}>
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={ACCENT} stopOpacity={0.3} />
                <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              domain={[0, 100]}
            />
            <Tooltip contentStyle={tooltipStyle} />
            <Area
              type="monotone"
              dataKey="score"
              stroke={ACCENT}
              fill="url(#scoreGradient)"
              strokeWidth={2}
              dot={{ r: 4, fill: ACCENT }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </motion.div>

      {/* ── Line Chart – Individual Metrics ──────────────────────────── */}
      <motion.div
        className="card"
        style={{ padding: 'var(--space-lg)', marginTop: 'var(--space-xl)' }}
        {...fadeUp}
        transition={{ delay: 0.2 }}
      >
        <h3 className="heading-4" style={{ marginBottom: 'var(--space-lg)' }}>
          Metric Breakdown
        </h3>
        <div
          style={{
            display: 'flex',
            gap: 'var(--space-lg)',
            marginBottom: 'var(--space-md)',
            flexWrap: 'wrap',
          }}
        >
          {[
            { color: '#2563EB', name: 'Angle Deviation' },
            { color: '#10B981', name: 'Trajectory Deviation' },
            { color: '#F59E0B', name: 'Velocity Difference' },
            { color: '#8B5CF6', name: 'Tool Alignment' },
          ].map((l) => (
            <div
              key={l.name}
              style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}
            >
              <div
                style={{
                  width: 12,
                  height: 3,
                  borderRadius: 2,
                  background: l.color,
                }}
              />
              <span className="text-small" style={{ color: 'var(--text-secondary)' }}>
                {l.name}
              </span>
            </div>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={progressData}>
            <XAxis
              dataKey="date"
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="var(--text-muted)"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip contentStyle={tooltipStyle} />
            <Line
              type="monotone"
              dataKey="metric1"
              name="Angle"
              stroke="#2563EB"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="metric2"
              name="Trajectory"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="metric3"
              name="Velocity"
              stroke="#F59E0B"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="metric4"
              name="Alignment"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* ── Radar Chart – Skill Breakdown ────────────────────────────── */}
      <motion.div
        className="card"
        style={{ padding: 'var(--space-lg)', marginTop: 'var(--space-xl)' }}
        {...fadeUp}
        transition={{ delay: 0.3 }}
      >
        <h3 className="heading-4" style={{ marginBottom: 'var(--space-lg)' }}>
          Skill Breakdown
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
            <PolarGrid stroke="var(--border-default)" />
            <PolarAngleAxis dataKey="metric" stroke="var(--text-secondary)" fontSize={13} />
            <Radar
              dataKey="value"
              stroke={ACCENT}
              fill={ACCENT}
              fillOpacity={0.2}
              strokeWidth={2}
            />
            <Tooltip contentStyle={tooltipStyle} />
          </RadarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* ── Course Progress ──────────────────────────────────────────── */}
      <div style={{ marginTop: 'var(--space-xl)' }}>
        <h3 className="heading-4" style={{ marginBottom: 'var(--space-lg)' }}>
          Course Progress
        </h3>
        <div className="grid-courses">
          {activeCourses.map((course, i) => (
            <motion.div
              key={course.id}
              className="card"
              style={{ padding: 'var(--space-lg)' }}
              {...fadeUp}
              transition={{ delay: 0.35 + i * 0.05 }}
            >
              <div className="flex-between">
                <h4 className="heading-4">{course.title}</h4>
                <span
                  className={`difficulty-badge difficulty-${course.difficulty}`}
                >
                  {course.difficulty}
                </span>
              </div>
              <p
                className="text-small"
                style={{ color: 'var(--text-secondary)', marginTop: 'var(--space-xs)' }}
              >
                {course.instructor} · {course.estimatedTime}
              </p>
              <div style={{ marginTop: 'var(--space-md)' }}>
                <div
                  className="flex-between"
                  style={{ marginBottom: 'var(--space-xs)' }}
                >
                  <span
                    className="text-small"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    Progress
                  </span>
                  <span className="text-small" style={{ fontWeight: 600 }}>
                    {course.progress}%
                  </span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${course.progress}%` }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
