import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Mail,
  CalendarDays,
  TrendingUp,
  Target,
  Flame,
  Trophy,
  Star,
  Edit3,
  Check,
  Github,
  Twitter,
  Linkedin,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { userProfile } from '@/services/mock/user';
import { achievements } from '@/services/mock/achievements';

const recentAchievements = achievements
  .filter((a) => a.unlockedAt)
  .sort((a, b) => new Date(b.unlockedAt!).getTime() - new Date(a.unlockedAt!).getTime())
  .slice(0, 4);

function getInitials(name: string) {
  return name
    .split(' ')
    .map((w) => w[0])
    .join('')
    .toUpperCase();
}

export default function ProfilePage() {
  const [bio, setBio] = useState(
    'Passionate learner exploring ceramics and woodcraft. Always looking to improve my technique through practice and AI-powered feedback.',
  );
  const [isEditingBio, setIsEditingBio] = useState(false);

  const xpPercent = Math.round((userProfile.xp / userProfile.nextLevelXp) * 100);
  const initials = getInitials(userProfile.name);

  return (
    <div className="container section">
      {/* ── Cover + Avatar ────────────────────────────────────────────── */}
      <motion.div
        style={{ position: 'relative', marginBottom: 'var(--space-3xl)' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Cover gradient */}
        <div
          style={{
            height: 200,
            borderRadius: 'var(--radius-xl)',
            background:
              'linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background:
                'radial-gradient(ellipse at 30% 50%, rgba(59,130,246,0.3), transparent 70%)',
            }}
          />
        </div>

        {/* Avatar */}
        <div
          style={{
            position: 'absolute',
            bottom: -48,
            left: 'var(--space-xl)',
            display: 'flex',
            alignItems: 'flex-end',
            gap: 'var(--space-lg)',
          }}
        >
          <div
            style={{
              width: 96,
              height: 96,
              borderRadius: '50%',
              background: 'var(--accent-primary)',
              border: '4px solid var(--bg-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1.75rem',
              fontWeight: 700,
              color: '#fff',
              boxShadow: 'var(--shadow-glow)',
            }}
          >
            {initials}
          </div>
        </div>
      </motion.div>

      {/* ── User Info ─────────────────────────────────────────────────── */}
      <motion.div
        style={{ paddingLeft: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="heading-2">{userProfile.name}</h2>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-lg)',
            marginTop: 'var(--space-sm)',
            flexWrap: 'wrap',
          }}
        >
          <span
            className="text-small"
            style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 4 }}
          >
            <Mail size={14} />
            {userProfile.email}
          </span>
          <span
            className="text-small"
            style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 4 }}
          >
            <CalendarDays size={14} />
            Joined {new Date(userProfile.joinedAt).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
          </span>
        </div>
      </motion.div>

      {/* ── XP Bar ────────────────────────────────────────────────────── */}
      <motion.div
        className="card"
        style={{ padding: 'var(--space-lg)', marginTop: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        <div className="flex-between" style={{ marginBottom: 'var(--space-sm)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <Trophy size={18} style={{ color: 'var(--accent-primary)' }} />
            <span style={{ fontWeight: 600 }}>Experience Points</span>
          </div>
          <span className="badge badge-blue">Level {userProfile.level}</span>
        </div>
        <div className="progress-bar" style={{ height: 12 }}>
          <div
            className="progress-bar-fill"
            style={{ width: `${xpPercent}%` }}
          />
        </div>
        <div
          className="flex-between"
          style={{ marginTop: 'var(--space-xs)' }}
        >
          <span className="text-small" style={{ color: 'var(--text-muted)' }}>
            {userProfile.xp} XP
          </span>
          <span className="text-small" style={{ color: 'var(--text-muted)' }}>
            {userProfile.nextLevelXp} XP
          </span>
        </div>
      </motion.div>

      {/* ── Stats Row ─────────────────────────────────────────────────── */}
      <div className="stats-grid" style={{ marginTop: 'var(--space-xl)' }}>
        {[
          { icon: TrendingUp, label: 'Evaluations', value: userProfile.totalEvaluations },
          { icon: Target, label: 'Avg. Score', value: userProfile.averageScore },
          { icon: Flame, label: 'Streak', value: `${userProfile.streakDays} days` },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              className="card stat-card"
              style={{ textAlign: 'center' }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.05 }}
            >
              <Icon size={24} style={{ color: 'var(--accent-primary)', marginBottom: 'var(--space-sm)' }} />
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </motion.div>
          );
        })}
      </div>

      {/* ── Recent Achievements ───────────────────────────────────────── */}
      <motion.div
        style={{ marginTop: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h3 className="heading-4" style={{ marginBottom: 'var(--space-lg)' }}>
          Recent Achievements
        </h3>
        <div className="stats-grid">
          {recentAchievements.map((a) => (
            <div
              key={a.id}
              className="card"
              style={{
                padding: 'var(--space-md)',
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-md)',
              }}
            >
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--accent-soft)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <Star size={18} style={{ color: 'var(--accent-primary)' }} />
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{a.title}</div>
                <div className="text-small" style={{ color: 'var(--text-muted)' }}>
                  {new Date(a.unlockedAt!).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* ── Bio ───────────────────────────────────────────────────────── */}
      <motion.div
        className="card"
        style={{ padding: 'var(--space-lg)', marginTop: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <div className="flex-between" style={{ marginBottom: 'var(--space-md)' }}>
          <h3 className="heading-4">Bio</h3>
          <button
            className="btn btn-ghost"
            style={{ padding: '0.375rem' }}
            onClick={() => {
              if (isEditingBio) setIsEditingBio(false);
              else setIsEditingBio(true);
            }}
          >
            {isEditingBio ? <Check size={16} /> : <Edit3 size={16} />}
          </button>
        </div>
        {isEditingBio ? (
          <textarea
            className="input"
            value={bio}
            onChange={(e) => setBio(e.target.value)}
            rows={4}
            style={{ resize: 'vertical', fontFamily: 'inherit' }}
          />
        ) : (
          <p className="text-body">{bio}</p>
        )}
      </motion.div>

      {/* ── Social Links ──────────────────────────────────────────────── */}
      <motion.div
        className="card"
        style={{ padding: 'var(--space-lg)', marginTop: 'var(--space-xl)' }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="heading-4" style={{ marginBottom: 'var(--space-md)' }}>
          Social Links
        </h3>
        <div style={{ display: 'flex', gap: 'var(--space-md)' }}>
          {[
            { icon: Github, label: 'GitHub' },
            { icon: Twitter, label: 'Twitter' },
            { icon: Linkedin, label: 'LinkedIn' },
          ].map(({ icon: Icon, label }) => (
            <Button key={label} variant="secondary" size="sm">
              <Icon size={14} />
              {label}
            </Button>
          ))}
        </div>
        <p className="text-small" style={{ color: 'var(--text-muted)', marginTop: 'var(--space-md)' }}>
          Connect your social accounts to share your progress
        </p>
      </motion.div>
    </div>
  );
}
