import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Search, SlidersHorizontal, BookOpen, PlayCircle } from 'lucide-react';
import { courses } from '@/services/mock/courses';
import type { Course } from '@/types';

type Difficulty = 'all' | Course['difficulty'];
type SortKey = 'title' | 'difficulty' | 'progress';

const difficulties: { value: Difficulty; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'beginner', label: 'Beginner' },
  { value: 'intermediate', label: 'Intermediate' },
  { value: 'advanced', label: 'Advanced' },
];

const difficultyOrder: Record<Course['difficulty'], number> = {
  beginner: 0,
  intermediate: 1,
  advanced: 2,
};

const cardVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.45, ease: 'easeOut' },
  }),
};

export default function Courses() {
  const [query, setQuery] = useState('');
  const [difficulty, setDifficulty] = useState<Difficulty>('all');
  const [sort, setSort] = useState<SortKey>('title');
  const [expertVideoUrl, setExpertVideoUrl] = useState<string | null>(null);
  const [expertVideoError, setExpertVideoError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadExpertVideo = async () => {
      try {
        const response = await fetch('/api/chapters/default/expert-video');
        if (!response.ok) {
          throw new Error('Failed to load default expert video');
        }

        const payload: { url: string } = await response.json();
        if (!cancelled) {
          setExpertVideoError(null);
          setExpertVideoUrl(payload.url);
        }
      } catch {
        if (!cancelled) {
          setExpertVideoUrl(null);
          setExpertVideoError('Expert video is unavailable. Start the AugMentor 2.0 backend on port 8001 and seed an expert video.');
        }
      }
    };

    void loadExpertVideo();

    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    let result = [...courses];

    if (query.trim()) {
      const q = query.toLowerCase();
      result = result.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.description.toLowerCase().includes(q) ||
          c.instructor.toLowerCase().includes(q) ||
          c.category.toLowerCase().includes(q),
      );
    }

    if (difficulty !== 'all') {
      result = result.filter((c) => c.difficulty === difficulty);
    }

    result.sort((a, b) => {
      if (sort === 'title') return a.title.localeCompare(b.title);
      if (sort === 'difficulty') return difficultyOrder[a.difficulty] - difficultyOrder[b.difficulty];
      return b.progress - a.progress;
    });

    return result;
  }, [query, difficulty, sort]);

  return (
    <div className="container" style={{ paddingBottom: 'var(--space-3xl)' }}>
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="page-header">
        <h1 className="page-title">Courses</h1>
        <p className="page-subtitle">Browse our collection of expert-led craft courses</p>
      </div>

      {/* ── Expert Demo Preview ─────────────────────────────────────────── */}
      <motion.div
        className="glass"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.1fr) minmax(280px, 0.9fr)',
          gap: 'var(--space-xl)',
          padding: 'var(--space-xl)',
          borderRadius: 'var(--radius-xl)',
          marginBottom: 'var(--space-xl)',
        }}
      >
        <div>
          <div
            className="badge badge-blue"
            style={{ display: 'inline-flex', marginBottom: 'var(--space-sm)' }}
          >
            Featured Expert Demo
          </div>
          <h2 className="heading-3" style={{ marginBottom: 'var(--space-sm)' }}>
            Watch an expert technique before choosing your course
          </h2>
          <p
            className="text-body"
            style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-lg)' }}
          >
            This preview uses the current expert video stored in the backend so learners can
            immediately see the kind of guided practice they will compare against later.
          </p>
          <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
            <Link to="/courses/pottery-wheel" className="btn btn-primary">
              <PlayCircle size={16} />
              Explore Pottery Course
            </Link>
            <Link to="/compare" className="btn btn-secondary">
              Open Compare Studio
            </Link>
            <Link to="/expert-videos" className="btn btn-ghost">
              Manage Expert Videos
            </Link>
          </div>
        </div>

        <div
          className="video-container"
          style={{
            aspectRatio: '16/9',
            background: 'var(--bg-tertiary)',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
          }}
        >
          {expertVideoUrl ? (
            <video
              key={expertVideoUrl}
              src={expertVideoUrl}
              controls
              preload="metadata"
              playsInline
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            />
          ) : (
            <div
              className="flex-center"
              style={{
                width: '100%',
                height: '100%',
                color: 'var(--text-muted)',
                padding: 'var(--space-lg)',
                textAlign: 'center',
              }}
            >
              {expertVideoError ?? 'Expert video preview is loading...'}
            </div>
          )}
        </div>
      </motion.div>

      {/* ── Search & Filters ────────────────────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
        {/* Search */}
        <div style={{ position: 'relative', maxWidth: 480 }}>
          <Search
            size={16}
            style={{
              position: 'absolute',
              left: '0.875rem',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-muted)',
              pointerEvents: 'none',
            }}
          />
          <input
            className="input"
            placeholder="Search courses, instructors, categories..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ paddingLeft: '2.5rem' }}
          />
        </div>

        {/* Filter row */}
        <div className="flex-between" style={{ flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
          <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
            {difficulties.map((d) => (
              <button
                key={d.value}
                className={`btn ${difficulty === d.value ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setDifficulty(d.value)}
                style={{ fontSize: '0.8125rem', padding: '0.375rem 0.875rem', borderRadius: '9999px' }}
              >
                {d.label}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <SlidersHorizontal size={14} style={{ color: 'var(--text-muted)' }} />
            <select
              className="input"
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              style={{ width: 'auto', minWidth: 140, cursor: 'pointer' }}
            >
              <option value="title">Sort by Title</option>
              <option value="difficulty">Sort by Difficulty</option>
              <option value="progress">Sort by Progress</option>
            </select>
          </div>
        </div>
      </div>

      {/* ── Course Grid ─────────────────────────────────────────────────── */}
      {filtered.length > 0 ? (
        <div className="grid-courses">
          {filtered.map((course, i) => (
            <motion.div
              key={course.id}
              custom={i}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
            >
              <Link
                to={`/courses/${course.id}`}
                className="card card-hover"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  textDecoration: 'none',
                  color: 'inherit',
                  height: '100%',
                }}
              >
                {/* Thumbnail */}
                <div style={{ position: 'relative', aspectRatio: '16/9', background: 'var(--bg-tertiary)', overflow: 'hidden' }}>
                  {course.id === 'pottery-wheel' && expertVideoUrl ? (
                    <video
                      src={expertVideoUrl}
                      muted
                      playsInline
                      preload="metadata"
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  ) : (
                    <img
                      src={course.thumbnail}
                      alt={course.title}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  )}
                  <span
                    className={`difficulty-badge difficulty-${course.difficulty}`}
                    style={{ position: 'absolute', top: 'var(--space-sm)', right: 'var(--space-sm)' }}
                  >
                    {course.difficulty}
                  </span>
                  {course.id === 'pottery-wheel' && expertVideoUrl && (
                    <span
                      className="badge badge-blue"
                      style={{ position: 'absolute', top: 'var(--space-sm)', left: 'var(--space-sm)' }}
                    >
                      Real Expert Video
                    </span>
                  )}
                </div>

                {/* Body */}
                <div style={{ padding: 'var(--space-md)', display: 'flex', flexDirection: 'column', flex: 1 }}>
                  <h4 className="heading-4">{course.title}</h4>
                  <p
                    className="text-body"
                    style={{
                      marginTop: 'var(--space-xs)',
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                      fontSize: '0.875rem',
                    }}
                  >
                    {course.description}
                  </p>

                  <div style={{ marginTop: 'auto', paddingTop: 'var(--space-md)' }}>
                    <div className="flex-between" style={{ marginBottom: 'var(--space-sm)' }}>
                      <span className="text-small" style={{ color: 'var(--text-secondary)' }}>
                        {course.instructor}
                      </span>
                      <span className="text-small" style={{ color: 'var(--text-muted)' }}>
                        {course.estimatedTime}
                      </span>
                    </div>

                    {course.progress > 0 && (
                      <div className="progress-bar">
                        <div className="progress-bar-fill" style={{ width: `${course.progress}%` }} />
                      </div>
                    )}
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      ) : (
        /* ── Empty State ──────────────────────────────────────────────── */
        <div className="empty-state">
          <BookOpen className="empty-state-icon" />
          <h3 className="empty-state-title">No courses found</h3>
          <p className="empty-state-description">
            Try adjusting your search or filter to find what you're looking for.
          </p>
        </div>
      )}
    </div>
  );
}
