import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Star,
  Flame,
  Trophy,
  Target,
  Crown,
  Layers,
  Calendar,
  MessageCircle,
  Check,
  Lock,
  type LucideIcon,
} from 'lucide-react';
import { achievements } from '@/services/mock/achievements';

/* ─── Icon mapping from emoji → Lucide ──────────────────────────────────── */

const iconMap: Record<string, LucideIcon> = {
  '🎯': Target,
  '⭐': Star,
  '🏆': Trophy,
  '🔥': Flame,
  '💪': Crown,
  '🎨': Layers,
  '💎': MessageCircle,
  '🏅': Calendar,
};

/* ─── Rarity colors ─────────────────────────────────────────────────────── */

const rarityBorder: Record<string, string> = {
  common: 'var(--text-muted)',
  rare: '#3B82F6',
  epic: '#8B5CF6',
  legendary: '#F59E0B',
};

const rarityBg: Record<string, string> = {
  common: 'rgba(100,116,139,0.1)',
  rare: 'rgba(59,130,246,0.1)',
  epic: 'rgba(139,92,246,0.1)',
  legendary: 'rgba(245,158,11,0.1)',
};

type StatusFilter = 'all' | 'unlocked' | 'locked';
type RarityFilter = 'all' | 'common' | 'rare' | 'epic' | 'legendary';

export default function AchievementsPage() {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [rarityFilter, setRarityFilter] = useState<RarityFilter>('all');

  const unlockedCount = achievements.filter((a) => a.unlockedAt).length;

  const filtered = useMemo(() => {
    return achievements.filter((a) => {
      if (statusFilter === 'unlocked' && !a.unlockedAt) return false;
      if (statusFilter === 'locked' && a.unlockedAt) return false;
      if (rarityFilter !== 'all' && a.rarity !== rarityFilter) return false;
      return true;
    });
  }, [statusFilter, rarityFilter]);

  return (
    <div className="container section">
      <div className="page-header">
        <div className="flex-between" style={{ flexWrap: 'wrap', gap: 'var(--space-md)' }}>
          <div>
            <h1 className="page-title">Achievements</h1>
            <p className="page-subtitle">
              {unlockedCount} of {achievements.length} unlocked
            </p>
          </div>
          <span className="badge badge-blue" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>
            {unlockedCount} / {achievements.length}
          </span>
        </div>
      </div>

      {/* ── Filters ───────────────────────────────────────────────────── */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 'var(--space-lg)',
          marginBottom: 'var(--space-xl)',
        }}
      >
        {/* Status tabs */}
        <div className="tabs" style={{ borderBottom: 'none', marginBottom: 0 }}>
          {(['all', 'unlocked', 'locked'] as StatusFilter[]).map((f) => (
            <button
              key={f}
              className={`tab ${statusFilter === f ? 'active' : ''}`}
              onClick={() => setStatusFilter(f)}
              style={{ background: 'none', border: 'none' }}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>

        {/* Rarity filter */}
        <div style={{ display: 'flex', gap: 'var(--space-xs)', alignItems: 'center' }}>
          {(['all', 'common', 'rare', 'epic', 'legendary'] as RarityFilter[]).map(
            (r) => (
              <button
                key={r}
                className={`btn ${rarityFilter === r ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setRarityFilter(r)}
                style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}
              >
                {r.charAt(0).toUpperCase() + r.slice(1)}
              </button>
            ),
          )}
        </div>
      </div>

      {/* ── Achievement Grid ──────────────────────────────────────────── */}
      <div className="grid-courses">
        {filtered.map((a, i) => {
          const isUnlocked = !!a.unlockedAt;
          const Icon = iconMap[a.icon] ?? Star;
          const borderColor = rarityBorder[a.rarity];
          const bgColor = rarityBg[a.rarity];

          return (
            <motion.div
              key={a.id}
              className="card"
              style={{
                padding: 'var(--space-lg)',
                borderColor,
                opacity: isUnlocked ? 1 : 0.55,
                position: 'relative',
                overflow: 'hidden',
              }}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: isUnlocked ? 1 : 0.55, scale: 1 }}
              transition={{ delay: i * 0.04 }}
            >
              {/* Rarity glow */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 3,
                  background: borderColor,
                }}
              />

              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 'var(--space-md)',
                }}
              >
                {/* Icon */}
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 'var(--radius-lg)',
                    background: bgColor,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <Icon size={22} style={{ color: borderColor }} />
                </div>

                {/* Content */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: 'var(--space-sm)',
                    }}
                  >
                    <h4 className="heading-4">{a.title}</h4>
                    {isUnlocked ? (
                      <div
                        style={{
                          width: 24,
                          height: 24,
                          borderRadius: '50%',
                          background: 'var(--success)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                        }}
                      >
                        <Check size={14} style={{ color: '#fff' }} />
                      </div>
                    ) : (
                      <Lock size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                    )}
                  </div>

                  <p
                    className="text-small"
                    style={{ color: 'var(--text-secondary)', marginTop: 'var(--space-xs)' }}
                  >
                    {a.description}
                  </p>

                  {/* Rarity label */}
                  <span
                    className="badge"
                    style={{
                      marginTop: 'var(--space-sm)',
                      background: bgColor,
                      color: borderColor,
                      textTransform: 'capitalize',
                    }}
                  >
                    {a.rarity}
                  </span>

                  {isUnlocked ? (
                    <div
                      className="text-small"
                      style={{ color: 'var(--text-muted)', marginTop: 'var(--space-sm)' }}
                    >
                      Unlocked{' '}
                      {new Date(a.unlockedAt!).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })}
                    </div>
                  ) : (
                    <div style={{ marginTop: 'var(--space-sm)' }}>
                      <div
                        className="flex-between"
                        style={{ marginBottom: 'var(--space-xs)' }}
                      >
                        <span className="text-small" style={{ color: 'var(--text-muted)' }}>
                          Progress
                        </span>
                        <span className="text-small" style={{ color: 'var(--text-secondary)' }}>
                          {a.progress}/{a.maxProgress}
                        </span>
                      </div>
                      <div className="progress-bar">
                        <div
                          className="progress-bar-fill"
                          style={{
                            width: `${(a.progress / a.maxProgress) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
