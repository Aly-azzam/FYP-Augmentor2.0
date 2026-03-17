import { useRef, useState, useEffect, Suspense } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { Float, MeshDistortMaterial, OrbitControls } from '@react-three/drei';
import { BookOpen, Play, Upload, Sparkles, ArrowRight, ChevronDown } from 'lucide-react';
import { courses } from '@/services/mock/courses';
import { useThemeStore } from '@/store';

/* ─── 3D Wireframe Vase ────────────────────────────────────────────────────── */

function WireframeVase({ color }: { color: string }) {
  return (
    <Float speed={1.4} rotationIntensity={1.2} floatIntensity={1.5}>
      <mesh>
        <cylinderGeometry args={[0.6, 1, 2.4, 32, 1, true]} />
        <MeshDistortMaterial
          color={color}
          wireframe
          distort={0.25}
          speed={2}
          transparent
          opacity={0.35}
        />
      </mesh>
    </Float>
  );
}

/* ─── Animated Counter ─────────────────────────────────────────────────────── */

function CountUp({ target, suffix = '' }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const triggered = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !triggered.current) {
          triggered.current = true;
          const duration = 1600;
          const start = performance.now();
          const step = (now: number) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCount(Math.floor(eased * target));
            if (progress < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [target]);

  return (
    <span ref={ref}>
      {count}
      {suffix}
    </span>
  );
}

/* ─── Constants ────────────────────────────────────────────────────────────── */

const stats = [
  { value: 500, suffix: '+', label: 'Active Learners' },
  { value: 50, suffix: '+', label: 'Expert Courses' },
  { value: 10000, suffix: '+', label: 'Evaluations Done' },
  { value: 95, suffix: '%', label: 'Satisfaction Rate' },
];

const steps = [
  { icon: BookOpen, title: 'Choose a Course', description: 'Browse our curated library of expert-led courses across various crafts and disciplines.' },
  { icon: Play, title: 'Watch Expert Videos', description: 'Study frame-by-frame expert demonstrations with detailed annotations and key points.' },
  { icon: Upload, title: 'Upload Your Practice', description: 'Record and upload your own practice attempts for side-by-side comparison.' },
  { icon: Sparkles, title: 'Get AI Feedback', description: 'Receive detailed AI-powered evaluation with metrics, scores, and actionable suggestions.' },
];

const fadeInUp = {
  initial: { opacity: 0, y: 40 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: '-80px' },
  transition: { duration: 0.6, ease: 'easeOut' },
};

const staggerContainer = {
  initial: {},
  whileInView: { transition: { staggerChildren: 0.15 } },
  viewport: { once: true, margin: '-80px' },
};

const staggerChild = {
  initial: { opacity: 0, y: 30 },
  whileInView: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

/* ─── Home Page ────────────────────────────────────────────────────────────── */

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);
  const isDark = useThemeStore((s) => s.isDark);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.92]);

  return (
    <>
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section ref={heroRef} style={{ height: '100vh', position: 'relative', overflow: 'hidden' }}>
        {/* Animated gradient background */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(37,99,235,0.18) 0%, transparent 70%), radial-gradient(ellipse 60% 50% at 80% 20%, rgba(59,130,246,0.12) 0%, transparent 60%)',
            animation: 'bgRotate 20s linear infinite',
          }}
        />
        <style>{`@keyframes bgRotate { 0%{filter:hue-rotate(0deg)} 100%{filter:hue-rotate(360deg)} }`}</style>

        <motion.div
          style={{ opacity: heroOpacity, scale: heroScale, height: '100%', position: 'relative', zIndex: 1 }}
        >
          <div className="container" style={{ height: '100%', display: 'flex', alignItems: 'center', gap: 'var(--space-2xl)' }}>
            {/* Left content */}
            <div style={{ flex: 1 }}>
              <motion.h1
                className="heading-1"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7 }}
              >
                Master Any Craft with{' '}
                <span className="gradient-text">AI-Powered Guidance</span>
              </motion.h1>

              <motion.p
                className="text-body"
                style={{ marginTop: 'var(--space-lg)', maxWidth: 540 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.2 }}
              >
                Upload your practice videos, compare with expert demonstrations, and receive
                detailed AI-powered feedback to accelerate your learning journey.
              </motion.p>

              <motion.div
                style={{ display: 'flex', gap: 'var(--space-md)', marginTop: 'var(--space-xl)' }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.4 }}
              >
                <Link to="/courses" className="btn btn-primary" style={{ gap: '0.5rem' }}>
                  Start Learning <ArrowRight size={16} />
                </Link>
                <Link to="/compare" className="btn btn-secondary">
                  Compare Studio
                </Link>
              </motion.div>
            </div>

            {/* Right 3D Canvas */}
            <motion.div
              style={{ flex: 1, height: '60%', minHeight: 340 }}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.3 }}
            >
              <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
                <ambientLight intensity={0.4} />
                <directionalLight position={[5, 5, 5]} intensity={0.6} />
                <Suspense fallback={null}>
                  <WireframeVase color={isDark ? '#FFFFFF' : '#3B82F6'} />
                </Suspense>
                <OrbitControls
                  enablePan={false}
                  enableZoom={false}
                  minPolarAngle={Math.PI / 3}
                  maxPolarAngle={(Math.PI * 2) / 3}
                />
              </Canvas>
            </motion.div>
          </div>

          {/* Scroll indicator */}
          <motion.div
            style={{
              position: 'absolute',
              bottom: 'var(--space-xl)',
              left: '50%',
              transform: 'translateX(-50%)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 'var(--space-xs)',
            }}
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
          >
            <span className="text-small" style={{ color: 'var(--text-muted)' }}>Scroll</span>
            <ChevronDown size={18} style={{ color: 'var(--text-muted)' }} />
          </motion.div>
        </motion.div>
      </section>

      {/* ── Stats ────────────────────────────────────────────────────────── */}
      <section className="section">
        <div className="container">
          <motion.div className="stats-grid" {...fadeInUp}>
            {stats.map((stat) => (
              <div className="card stat-card" key={stat.label} style={{ textAlign: 'center' }}>
                <div className="stat-value">
                  <CountUp
                    target={stat.value}
                    suffix={stat.suffix}
                  />
                </div>
                <div className="stat-label">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── Featured Courses ─────────────────────────────────────────────── */}
      <section className="section">
        <div className="container">
          <motion.h2 className="heading-2" style={{ marginBottom: 'var(--space-xl)' }} {...fadeInUp}>
            Featured Courses
          </motion.h2>
        </div>
        <div className="container container-wide">
          <motion.div className="featured-marquee" style={{ paddingBottom: 'var(--space-md)' }} {...fadeInUp}>
            <div className="featured-marquee-track">
              {[...courses, ...courses].map((course, idx) => (
                <Link
                  to={`/courses/${course.id}`}
                  key={`${course.id}-${idx}`}
                  className="card card-hover"
                  style={{
                    width: 320,
                    textDecoration: 'none',
                    color: 'inherit',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      height: 180,
                      background: 'var(--bg-tertiary)',
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    <img
                      src={course.thumbnail}
                      alt={course.title}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                    <span
                      className={`difficulty-badge difficulty-${course.difficulty}`}
                      style={{ position: 'absolute', top: 'var(--space-sm)', right: 'var(--space-sm)' }}
                    >
                      {course.difficulty}
                    </span>
                  </div>
                  <div style={{ padding: 'var(--space-md)' }}>
                    <h4 className="heading-4">{course.title}</h4>
                    <p className="text-small" style={{ color: 'var(--text-secondary)', marginTop: 'var(--space-xs)' }}>
                      {course.instructor}
                    </p>
                    {course.progress > 0 && (
                      <div style={{ marginTop: 'var(--space-sm)' }}>
                        <div className="progress-bar">
                          <div className="progress-bar-fill" style={{ width: `${course.progress}%` }} />
                        </div>
                      </div>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── How It Works ─────────────────────────────────────────────────── */}
      <section className="section" style={{ background: 'var(--bg-secondary)' }}>
        <div className="container" style={{ textAlign: 'center' }}>
          <motion.h2 className="heading-2" style={{ marginBottom: 'var(--space-3xl)' }} {...fadeInUp}>
            How It Works
          </motion.h2>

          <motion.div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: 'var(--space-xl)',
            }}
            {...staggerContainer}
          >
            {steps.map((step, i) => {
              const Icon = step.icon;
              return (
                <motion.div key={step.title} style={{ textAlign: 'center' }} {...staggerChild}>
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: '50%',
                      background: 'var(--accent-soft)',
                      color: 'var(--accent-primary)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto var(--space-sm)',
                      fontSize: '0.875rem',
                      fontWeight: 700,
                    }}
                  >
                    {i + 1}
                  </div>
                  <div
                    style={{
                      width: 56,
                      height: 56,
                      borderRadius: 'var(--radius-lg)',
                      background: 'var(--bg-tertiary)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto var(--space-md)',
                    }}
                  >
                    <Icon size={24} style={{ color: 'var(--accent-glow)' }} />
                  </div>
                  <h4 className="heading-4" style={{ marginBottom: 'var(--space-sm)' }}>{step.title}</h4>
                  <p className="text-body" style={{ fontSize: '0.875rem' }}>{step.description}</p>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────────── */}
      <section className="section">
        <div className="container" style={{ textAlign: 'center' }}>
          <motion.div {...fadeInUp}>
            <h2 className="heading-2" style={{ marginBottom: 'var(--space-lg)' }}>
              Ready to improve your craft?
            </h2>
            <Link to="/courses" className="btn btn-primary" style={{ gap: '0.5rem' }}>
              Start Learning <ArrowRight size={16} />
            </Link>
          </motion.div>
        </div>
      </section>
    </>
  );
}
