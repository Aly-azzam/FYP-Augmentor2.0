import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Edit2, Mail, Calendar, Github, Twitter, Linkedin, Save, X } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface EvalStats {
  count: number;
  avgScore: number | null;
}

export default function ProfilePage() {
  const { user, token } = useAuth();
  const [stats, setStats] = useState<EvalStats>({ count: 0, avgScore: null });
  const [editingBio, setEditingBio] = useState(false);
  const [editingLinks, setEditingLinks] = useState(false);
  const [bio, setBio] = useState(user?.bio || '');
  const [github, setGithub] = useState(user?.github_url || '');
  const [twitter, setTwitter] = useState(user?.twitter_url || '');
  const [linkedin, setLinkedin] = useState(user?.linkedin_url || '');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!user || !token) return;
    setBio(user.bio || '');
    setGithub(user.github_url || '');
    setTwitter(user.twitter_url || '');
    setLinkedin(user.linkedin_url || '');
  }, [user]);

  useEffect(() => {
    if (!user || !token) return;
    fetch(`${API}/api/history?user_id=${user.id}&limit=200`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : [])
      .then((entries: { score: number | null }[]) => {
        const scored = entries.filter(e => e.score !== null);
        const avg = scored.length > 0
          ? Math.round(scored.reduce((sum, e) => sum + (e.score ?? 0), 0) / scored.length)
          : null;
        setStats({ count: entries.length, avgScore: avg });
      })
      .catch(() => {});
  }, [user, token]);

  const saveProfile = async (fields: object) => {
    if (!token) return;
    setSaving(true);
    try {
      await fetch(`${API}/api/auth/profile`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(fields),
      });
    } finally {
      setSaving(false);
    }
  };

  if (!user) return null;

  const initials = user.display_name?.split(' ').map((w: string) => w[0]).join('').slice(0, 2).toUpperCase() || '?';
  const joinedDate = user.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
    : null;

  return (
    <div className="container section" style={{ maxWidth: 800 }}>
      {/* Header banner */}
      <div style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #3b6fd4 100%)', borderRadius: 'var(--radius-lg)', height: 140, marginBottom: 0 }} />

      {/* Avatar + name */}
      <div style={{ padding: '0 var(--space-xl)', marginTop: -40, marginBottom: 'var(--space-xl)' }}>
        <div style={{ width: 80, height: 80, borderRadius: '50%', background: 'var(--accent-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.75rem', fontWeight: 700, color: '#fff', border: '4px solid var(--bg-primary)', marginBottom: 'var(--space-md)' }}>
          {initials}
        </div>
        <h1 className="heading-2" style={{ marginBottom: 4 }}>{user.display_name}</h1>
        <div style={{ display: 'flex', gap: 'var(--space-lg)', flexWrap: 'wrap' }}>
          <span className="text-small" style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Mail size={13} /> {user.email}
          </span>
          {joinedDate && (
            <span className="text-small" style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <Calendar size={13} /> Joined {joinedDate}
            </span>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="stat-card">
          <div className="stat-label">Evaluations</div>
          <div className="stat-value">{stats.count}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg. Score</div>
          <div className="stat-value">{stats.avgScore !== null ? `${stats.avgScore}%` : '—'}</div>
        </div>
      </div>

      {/* Bio */}
      <div className="card" style={{ padding: 'var(--space-lg)', marginBottom: 'var(--space-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
          <span className="heading-4">Bio</span>
          {!editingBio ? (
            <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={() => setEditingBio(true)}>
              <Edit2 size={14} />
            </button>
          ) : (
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={() => { setEditingBio(false); setBio(user.bio || ''); }}>
                <X size={14} />
              </button>
              <button className="btn btn-primary" style={{ padding: '4px 12px', fontSize: '0.8rem' }} disabled={saving}
                onClick={async () => { await saveProfile({ bio }); setEditingBio(false); }}>
                <Save size={14} /> {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          )}
        </div>
        {!editingBio ? (
          <p className="text-body" style={{ color: bio ? 'var(--text-secondary)' : 'var(--text-muted)', fontStyle: bio ? 'normal' : 'italic' }}>
            {bio || 'No bio yet. Click the edit button to add one.'}
          </p>
        ) : (
          <textarea
            className="input"
            value={bio}
            onChange={e => setBio(e.target.value)}
            rows={4}
            placeholder="Write something about yourself…"
            style={{ width: '100%', resize: 'vertical' }}
          />
        )}
      </div>

      {/* Social Links */}
      <div className="card" style={{ padding: 'var(--space-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-md)' }}>
          <span className="heading-4">Social Links</span>
          {!editingLinks ? (
            <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={() => setEditingLinks(true)}>
              <Edit2 size={14} />
            </button>
          ) : (
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={() => { setEditingLinks(false); }}>
                <X size={14} />
              </button>
              <button className="btn btn-primary" style={{ padding: '4px 12px', fontSize: '0.8rem' }} disabled={saving}
                onClick={async () => { await saveProfile({ github_url: github, twitter_url: twitter, linkedin_url: linkedin }); setEditingLinks(false); }}>
                <Save size={14} /> {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          )}
        </div>
        {!editingLinks ? (
          <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
            {github && <a href={github} target="_blank" rel="noreferrer" className="btn btn-ghost" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.875rem' }}><Github size={15} /> GitHub</a>}
            {twitter && <a href={twitter} target="_blank" rel="noreferrer" className="btn btn-ghost" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.875rem' }}><Twitter size={15} /> Twitter</a>}
            {linkedin && <a href={linkedin} target="_blank" rel="noreferrer" className="btn btn-ghost" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.875rem' }}><Linkedin size={15} /> LinkedIn</a>}
            {!github && !twitter && !linkedin && <p className="text-small" style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>No social links yet. Click edit to add some.</p>}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
              <Github size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              <input className="input" value={github} onChange={e => setGithub(e.target.value)} placeholder="https://github.com/username" style={{ flex: 1 }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
              <Twitter size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              <input className="input" value={twitter} onChange={e => setTwitter(e.target.value)} placeholder="https://twitter.com/username" style={{ flex: 1 }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
              <Linkedin size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              <input className="input" value={linkedin} onChange={e => setLinkedin(e.target.value)} placeholder="https://linkedin.com/in/username" style={{ flex: 1 }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
