import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Search, BookOpen, ImagePlus } from 'lucide-react';

type BackendCourse = {
  id: string;
  title: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  thumbnail_url?: string | null;
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
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [courses, setCourses] = useState<BackendCourse[]>([]);
  const [coursesLoading, setCoursesLoading] = useState(true);
  const [coursesError, setCoursesError] = useState<string | null>(null);
  const [newCourseTitle, setNewCourseTitle] = useState('');
  const [newCourseDescription, setNewCourseDescription] = useState('');
  const [creatingCourse, setCreatingCourse] = useState(false);
  const [deletingCourseId, setDeletingCourseId] = useState<string | null>(null);
  const [uploadingCourseImageId, setUploadingCourseImageId] = useState<string | null>(null);

  const loadCourses = useCallback(async () => {
    setCoursesLoading(true);
    setCoursesError(null);
    try {
      const response = await fetch('/api/courses');
      if (!response.ok) {
        throw new Error('Failed to load courses');
      }

      const payload = (await response.json()) as BackendCourse[];
      setCourses(payload);
    } catch {
      setCourses([]);
      setCoursesError('Could not load courses. Make sure the backend is running.');
    } finally {
      setCoursesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCourses();
  }, [loadCourses]);

  const handleCreateCourse = async () => {
    const title = newCourseTitle.trim();
    if (!title) {
      setCoursesError('Course title is required.');
      return;
    }

    setCreatingCourse(true);
    setCoursesError(null);
    try {
      const response = await fetch('/api/courses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title,
          description: newCourseDescription.trim() || null,
        }),
      });
      if (!response.ok) {
        let message = 'Failed to create course.';
        try {
          const payload = (await response.json()) as { detail?: string };
          message = payload.detail || message;
        } catch {
          // Keep generic message if response is not JSON.
        }
        throw new Error(message);
      }

      setNewCourseTitle('');
      setNewCourseDescription('');
      await loadCourses();
    } catch (error) {
      setCoursesError(error instanceof Error ? error.message : 'Failed to create course.');
    } finally {
      setCreatingCourse(false);
    }
  };

  const handleDeleteCourse = async (course: BackendCourse) => {
    const confirmed = window.confirm(`Delete course "${course.title}"?`);
    if (!confirmed) {
      return;
    }

    setDeletingCourseId(course.id);
    setCoursesError(null);
    try {
      const response = await fetch(`/api/courses/${encodeURIComponent(course.id)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        let message = 'Failed to delete course.';
        try {
          const payload = (await response.json()) as { detail?: string };
          message = payload.detail || message;
        } catch {
          // Keep generic message if response is not JSON.
        }
        throw new Error(message);
      }

      await loadCourses();
    } catch (error) {
      setCoursesError(error instanceof Error ? error.message : 'Failed to delete course.');
    } finally {
      setDeletingCourseId(null);
    }
  };

  const handleCourseImageUpload = async (course: BackendCourse, file: File | null) => {
    if (!file) return;

    setUploadingCourseImageId(course.id);
    setCoursesError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`/api/courses/${encodeURIComponent(course.id)}/thumbnail`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let message = response.status === 404
          ? 'Course image upload endpoint was not found. Restart the backend server and try again.'
          : 'Failed to upload course image.';
        try {
          const payload = (await response.json()) as { detail?: string };
          message = response.status === 404 ? message : payload.detail || message;
        } catch {
          // Keep generic message if response is not JSON.
        }
        throw new Error(message);
      }

      await loadCourses();
    } catch (error) {
      setCoursesError(error instanceof Error ? error.message : 'Failed to upload course image.');
    } finally {
      setUploadingCourseImageId(null);
    }
  };

  const filtered = useMemo(() => {
    let result = [...courses];

    if (query.trim()) {
      const q = query.toLowerCase();
      result = result.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          (c.description ?? '').toLowerCase().includes(q),
      );
    }

    result.sort((a, b) => a.title.localeCompare(b.title));

    return result;
  }, [courses, query]);

  return (
    <div className="container" style={{ paddingBottom: 'var(--space-3xl)' }}>
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="page-header">
        <h1 className="page-title">Courses</h1>
        <p className="page-subtitle">Browse our collection of expert-led craft courses</p>
      </div>

      {/* ── Create Course ───────────────────────────────────────────────── */}
      <div
        className="glass"
        style={{
          padding: 'var(--space-lg)',
          borderRadius: 'var(--radius-xl)',
          marginBottom: 'var(--space-xl)',
        }}
      >
        <h2 className="heading-4" style={{ marginBottom: 'var(--space-sm)' }}>
          Create Course
        </h2>
        <div style={{ display: 'grid', gap: 'var(--space-sm)' }}>
          <input
            className="input"
            value={newCourseTitle}
            onChange={(event) => setNewCourseTitle(event.target.value)}
            placeholder="Course title"
          />
          <textarea
            className="input"
            value={newCourseDescription}
            onChange={(event) => setNewCourseDescription(event.target.value)}
            placeholder="Course description"
            rows={3}
            style={{ resize: 'vertical' }}
          />
          <div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void handleCreateCourse()}
              disabled={creatingCourse}
            >
              {creatingCourse ? 'Creating...' : 'Create Course'}
            </button>
          </div>
        </div>
      </div>

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
            placeholder="Search courses..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ paddingLeft: '2.5rem' }}
          />
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
              <div
                className="card card-hover"
                role="button"
                tabIndex={0}
                onClick={() => navigate(`/courses/${course.id}`)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    navigate(`/courses/${course.id}`);
                  }
                }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  color: 'inherit',
                  height: '100%',
                  cursor: 'pointer',
                }}
              >
                {/* Header */}
                <div
                  className="flex-center"
                  style={{
                    aspectRatio: '16/9',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-muted)',
                    overflow: 'hidden',
                  }}
                >
                  {course.thumbnail_url ? (
                    <img
                      src={course.thumbnail_url}
                      alt=""
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                  ) : (
                    <BookOpen size={40} />
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
                    {course.description || 'No description has been added for this course yet.'}
                  </p>

                  <div style={{ marginTop: 'auto', paddingTop: 'var(--space-md)' }}>
                    <div className="flex-between" style={{ marginBottom: 'var(--space-sm)' }}>
                      <span className="text-small" style={{ color: 'var(--text-secondary)' }}>
                        Created
                      </span>
                      <span className="text-small" style={{ color: 'var(--text-muted)' }}>
                        {new Date(course.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
                      <Link
                        to={`/courses/${course.id}`}
                        className="btn btn-secondary"
                        onClick={(event) => event.stopPropagation()}
                      >
                        View Course
                      </Link>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        disabled={uploadingCourseImageId === course.id}
                        onClick={(event) => {
                          event.stopPropagation();
                          document.getElementById(`course-image-${course.id}`)?.click();
                        }}
                      >
                        <ImagePlus size={16} />
                        {uploadingCourseImageId === course.id
                          ? 'Uploading...'
                          : course.thumbnail_url
                            ? 'Change Photo'
                            : 'Add Photo'}
                      </button>
                      <input
                        id={`course-image-${course.id}`}
                        type="file"
                        accept="image/jpeg,image/png,image/webp,image/gif"
                        disabled={uploadingCourseImageId === course.id}
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) => {
                          event.stopPropagation();
                          void handleCourseImageUpload(course, event.target.files?.[0] ?? null);
                          event.currentTarget.value = '';
                        }}
                        style={{ display: 'none' }}
                      />
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleDeleteCourse(course);
                        }}
                        disabled={deletingCourseId === course.id}
                      >
                        {deletingCourseId === course.id ? 'Deleting...' : 'Delete Course'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        /* ── Empty State ──────────────────────────────────────────────── */
        <div className="empty-state">
          <BookOpen className="empty-state-icon" />
          <h3 className="empty-state-title">
            {coursesLoading ? 'Loading courses...' : 'No courses found'}
          </h3>
          <p className="empty-state-description">
            {coursesError ?? "Try adjusting your search or filter to find what you're looking for."}
          </p>
        </div>
      )}
    </div>
  );
}
