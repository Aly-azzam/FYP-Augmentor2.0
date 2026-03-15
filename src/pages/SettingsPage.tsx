import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Palette,
  Bell,
  Play,
  Shield,
  User,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
} from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useThemeStore } from '@/store';

const ACCENT_OPTIONS = [
  { color: '#2563EB', label: 'Blue' },
  { color: '#8B5CF6', label: 'Purple' },
  { color: '#10B981', label: 'Emerald' },
  { color: '#F59E0B', label: 'Amber' },
  { color: '#EF4444', label: 'Red' },
  { color: '#EC4899', label: 'Pink' },
];

const FAQ_ITEMS = [
  {
    q: 'How does the AI evaluation work?',
    a: 'Our AI compares your practice video with expert demonstrations using computer vision. It measures angle deviation, trajectory, velocity, and tool alignment to provide a comprehensive score.',
  },
  {
    q: 'Can I upload videos from my phone?',
    a: 'Yes! You can upload videos in MP4, MOV, or WebM format from any device. For best results, ensure good lighting and a stable camera angle.',
  },
  {
    q: 'How is my score calculated?',
    a: 'Your score is a weighted average of four metrics: angle deviation, trajectory deviation, velocity difference, and tool alignment. Lower deviations result in higher scores.',
  },
  {
    q: 'What are streaks?',
    a: 'Streaks track consecutive days of practice. Complete at least one evaluation per day to maintain your streak and earn achievements.',
  },
];

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="flex-between"
      style={{
        padding: 'var(--space-md) 0',
        borderBottom: '1px solid var(--border-default)',
        gap: 'var(--space-lg)',
      }}
    >
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 500 }}>{label}</div>
        {description && (
          <div className="text-small" style={{ color: 'var(--text-secondary)', marginTop: 2 }}>
            {description}
          </div>
        )}
      </div>
      {children}
    </div>
  );
}

export default function SettingsPage() {
  const { isDark, toggleTheme } = useThemeStore();

  const [accentColor, setAccentColor] = useState('#2563EB');
  const [fontSize, setFontSize] = useState([16]);

  const [emailNotif, setEmailNotif] = useState(true);
  const [pushNotif, setPushNotif] = useState(true);
  const [weeklyDigest, setWeeklyDigest] = useState(false);

  const [playbackSpeed, setPlaybackSpeed] = useState([1]);
  const [autoPlay, setAutoPlay] = useState(true);
  const [showTimeline, setShowTimeline] = useState(true);

  const [profileVisible, setProfileVisible] = useState(true);
  const [showProgress, setShowProgress] = useState(true);
  const [anonymousMode, setAnonymousMode] = useState(false);

  const [name, setName] = useState('Alex Rivera');
  const [email, setEmail] = useState('alex.rivera@example.com');

  const [openFaq, setOpenFaq] = useState<number | null>(null);

  const tabIcons = [
    { value: 'appearance', icon: Palette, label: 'Appearance' },
    { value: 'notifications', icon: Bell, label: 'Notifications' },
    { value: 'playback', icon: Play, label: 'Playback' },
    { value: 'privacy', icon: Shield, label: 'Privacy' },
    { value: 'account', icon: User, label: 'Account' },
    { value: 'help', icon: HelpCircle, label: 'Help' },
  ];

  return (
    <div className="container section">
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">Manage your preferences and account</p>
      </div>

      <Tabs defaultValue="appearance">
        <TabsList>
          {tabIcons.map(({ value, icon: Icon, label }) => (
            <TabsTrigger key={value} value={value}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem' }}>
                <Icon size={16} />
                {label}
              </span>
            </TabsTrigger>
          ))}
        </TabsList>

        {/* ── Appearance ──────────────────────────────────────────────── */}
        <TabsContent value="appearance">
          <motion.div
            className="card"
            style={{ padding: 'var(--space-xl)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <SettingRow label="Dark Mode" description="Toggle between dark and light theme">
              <Switch checked={isDark} onCheckedChange={toggleTheme} />
            </SettingRow>

            <SettingRow label="Accent Color" description="Choose your preferred accent color">
              <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                {ACCENT_OPTIONS.map((opt) => (
                  <button
                    key={opt.color}
                    onClick={() => setAccentColor(opt.color)}
                    title={opt.label}
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: '50%',
                      background: opt.color,
                      border:
                        accentColor === opt.color
                          ? '3px solid var(--text-primary)'
                          : '2px solid transparent',
                      cursor: 'pointer',
                      transition: 'border var(--transition-fast)',
                    }}
                  />
                ))}
              </div>
            </SettingRow>

            <SettingRow label="Font Size" description={`${fontSize[0]}px`}>
              <div style={{ width: 160 }}>
                <Slider
                  value={fontSize}
                  onValueChange={setFontSize}
                  min={12}
                  max={20}
                  step={1}
                />
              </div>
            </SettingRow>

            <div style={{ marginTop: 'var(--space-xl)', textAlign: 'right' }}>
              <Button>Save Changes</Button>
            </div>
          </motion.div>
        </TabsContent>

        {/* ── Notifications ───────────────────────────────────────────── */}
        <TabsContent value="notifications">
          <motion.div
            className="card"
            style={{ padding: 'var(--space-xl)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <SettingRow label="Email Notifications" description="Receive evaluation results via email">
              <Switch checked={emailNotif} onCheckedChange={setEmailNotif} />
            </SettingRow>
            <SettingRow label="Push Notifications" description="Browser push notifications for new content">
              <Switch checked={pushNotif} onCheckedChange={setPushNotif} />
            </SettingRow>
            <SettingRow label="Weekly Digest" description="Summary of your weekly progress">
              <Switch checked={weeklyDigest} onCheckedChange={setWeeklyDigest} />
            </SettingRow>
            <div style={{ marginTop: 'var(--space-xl)', textAlign: 'right' }}>
              <Button>Save Changes</Button>
            </div>
          </motion.div>
        </TabsContent>

        {/* ── Playback ────────────────────────────────────────────────── */}
        <TabsContent value="playback">
          <motion.div
            className="card"
            style={{ padding: 'var(--space-xl)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <SettingRow label="Default Playback Speed" description={`${playbackSpeed[0]}x`}>
              <div style={{ width: 160 }}>
                <Slider
                  value={playbackSpeed}
                  onValueChange={setPlaybackSpeed}
                  min={0.25}
                  max={2}
                  step={0.25}
                />
              </div>
            </SettingRow>
            <SettingRow label="Auto-Play" description="Automatically play next clip">
              <Switch checked={autoPlay} onCheckedChange={setAutoPlay} />
            </SettingRow>
            <SettingRow label="Show Timeline" description="Display video timeline bar">
              <Switch checked={showTimeline} onCheckedChange={setShowTimeline} />
            </SettingRow>
            <div style={{ marginTop: 'var(--space-xl)', textAlign: 'right' }}>
              <Button>Save Changes</Button>
            </div>
          </motion.div>
        </TabsContent>

        {/* ── Privacy ─────────────────────────────────────────────────── */}
        <TabsContent value="privacy">
          <motion.div
            className="card"
            style={{ padding: 'var(--space-xl)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <SettingRow label="Profile Visibility" description="Make your profile visible to other learners">
              <Switch checked={profileVisible} onCheckedChange={setProfileVisible} />
            </SettingRow>
            <SettingRow label="Show Progress" description="Display your progress on public profile">
              <Switch checked={showProgress} onCheckedChange={setShowProgress} />
            </SettingRow>
            <SettingRow label="Anonymous Mode" description="Hide your name in leaderboards">
              <Switch checked={anonymousMode} onCheckedChange={setAnonymousMode} />
            </SettingRow>
            <div style={{ marginTop: 'var(--space-xl)', textAlign: 'right' }}>
              <Button>Save Changes</Button>
            </div>
          </motion.div>
        </TabsContent>

        {/* ── Account ─────────────────────────────────────────────────── */}
        <TabsContent value="account">
          <motion.div
            className="card"
            style={{ padding: 'var(--space-xl)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
              <div>
                <label className="label" style={{ display: 'block', marginBottom: 'var(--space-sm)' }}>
                  Full Name
                </label>
                <Input value={name} onChange={(e) => setName(e.target.value)} />
              </div>

              <div>
                <label className="label" style={{ display: 'block', marginBottom: 'var(--space-sm)' }}>
                  Email Address
                </label>
                <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>

              <div className="divider" />

              <div className="flex-between">
                <div>
                  <div style={{ fontWeight: 500 }}>Change Password</div>
                  <div className="text-small" style={{ color: 'var(--text-secondary)' }}>
                    Update your account password
                  </div>
                </div>
                <Button variant="secondary">Change Password</Button>
              </div>

              <div className="divider" />

              <div className="flex-between">
                <div>
                  <div style={{ fontWeight: 500, color: 'var(--error)' }}>Delete Account</div>
                  <div className="text-small" style={{ color: 'var(--text-secondary)' }}>
                    Permanently delete your account and all data
                  </div>
                </div>
                <Button variant="destructive">Delete Account</Button>
              </div>
            </div>

            <div style={{ marginTop: 'var(--space-xl)', textAlign: 'right' }}>
              <Button>Save Changes</Button>
            </div>
          </motion.div>
        </TabsContent>

        {/* ── Help ────────────────────────────────────────────────────── */}
        <TabsContent value="help">
          <motion.div
            className="card"
            style={{ padding: 'var(--space-xl)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <h3 className="heading-4" style={{ marginBottom: 'var(--space-lg)' }}>
              Frequently Asked Questions
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              {FAQ_ITEMS.map((item, i) => {
                const isOpen = openFaq === i;
                return (
                  <div
                    key={i}
                    className="glass"
                    style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden' }}
                  >
                    <button
                      onClick={() => setOpenFaq(isOpen ? null : i)}
                      style={{
                        width: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: 'var(--space-md)',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--text-primary)',
                        fontWeight: 500,
                        fontSize: '0.875rem',
                        textAlign: 'left',
                      }}
                    >
                      {item.q}
                      {isOpen ? (
                        <ChevronUp size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                      ) : (
                        <ChevronDown size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                      )}
                    </button>
                    {isOpen && (
                      <div
                        style={{
                          padding: '0 var(--space-md) var(--space-md)',
                          color: 'var(--text-secondary)',
                          fontSize: '0.875rem',
                          lineHeight: 1.7,
                        }}
                      >
                        {item.a}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="divider" />

            <div className="flex-between">
              <div>
                <div style={{ fontWeight: 500 }}>Contact Support</div>
                <div className="text-small" style={{ color: 'var(--text-secondary)' }}>
                  Need help? Reach out to our support team
                </div>
              </div>
              <a
                href="mailto:support@augmentor.app"
                className="btn btn-secondary"
                style={{ textDecoration: 'none', gap: '0.375rem' }}
              >
                <ExternalLink size={14} />
                Contact
              </a>
            </div>

            <div className="divider" />

            <div
              className="text-small"
              style={{ color: 'var(--text-muted)', textAlign: 'center' }}
            >
              AugMentor v2.0.0 · Built with ♥ for learners everywhere
            </div>
          </motion.div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
