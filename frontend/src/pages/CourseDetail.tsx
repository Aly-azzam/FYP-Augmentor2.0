import { useCallback, useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ChevronRight,
  AlertCircle,
  BookOpen,
  Play,
  Plus,
  Trash2,
  X,
} from 'lucide-react';
import { useCourseStore, useUIStore } from '../store';
import {
  createBackendChapter,
  deleteBackendChapter,
  fetchBackendChapters,
  fetchBackendCourseDetail,
  type BackendChapter,
  type BackendCourseDetail,
} from '../services/api/courses';

export default function CourseDetail() {
  const { courseId } = useParams<{ courseId: string }>();
  const navigate = useNavigate();
  const setSelectedCourse = useCourseStore((s) => s.setSelectedCourse);
  const setSelectedClip = useCourseStore((s) => s.setSelectedClip);
  const setRobotMessage = useUIStore((s) => s.setRobotMessage);
  const [course, setCourse] = useState<BackendCourseDetail | null>(null);
  const [chapters, setChapters] = useState<BackendChapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showChapterForm, setShowChapterForm] = useState(false);
  const [newChapterTitle, setNewChapterTitle] = useState('');
  const [newChapterDescription, setNewChapterDescription] = useState('');
  const [creatingChapter, setCreatingChapter] = useState(false);
  const [deletingChapterId, setDeletingChapterId] = useState<string | null>(null);

  useEffect(() => {
    if (course) {
      setSelectedCourse(course.id);
      setRobotMessage(
        `You're viewing "${course.title}". Pick a chapter with an expert video to compare.`,
      );
    }
    return () => {
      setRobotMessage(null);
    };
  }, [course, setSelectedCourse, setRobotMessage]);

  const loadCourse = useCallback(async (showLoading = true) => {
    if (!courseId) {
      setLoading(false);
      return;
    }

    if (showLoading) {
      setLoading(true);
    }
    setError(null);
    try {
      const [backendCourse, backendChapters] = await Promise.all([
        fetchBackendCourseDetail(courseId),
        fetchBackendChapters(courseId),
      ]);
      setCourse(backendCourse);
      setChapters(backendChapters);
    } catch {
      setCourse(null);
      setChapters([]);
      setError('Could not load course details. Make sure the backend is running.');
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, [courseId]);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      await loadCourse();
      if (cancelled) return;
    };

    void run();

    return () => {
      cancelled = true;
    };
  }, [loadCourse]);

  const openCompareStudio = (chapter: BackendChapter) => {
    if (!courseId || !chapter.expert_video?.url) return;

    setSelectedCourse(courseId);
    setSelectedClip(chapter.id);
    navigate(`/compare?courseId=${encodeURIComponent(courseId)}&clipId=${encodeURIComponent(chapter.id)}`);
  };

  const handleCreateChapter = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!courseId) return;

    const title = newChapterTitle.trim();
    if (!title) {
      setError('Chapter title is required.');
      return;
    }

    setCreatingChapter(true);
    setError(null);
    try {
      await createBackendChapter({
        course_id: courseId,
        title,
        description: newChapterDescription.trim() || null,
      });
      setNewChapterTitle('');
      setNewChapterDescription('');
      setShowChapterForm(false);
      await loadCourse(false);
    } catch {
      setError('Failed to create chapter.');
    } finally {
      setCreatingChapter(false);
    }
  };

  const handleDeleteChapter = async (chapter: BackendChapter) => {
    const confirmed = window.confirm(`Delete chapter "${chapter.title}"?`);
    if (!confirmed) return;

    setDeletingChapterId(chapter.id);
    setError(null);
    try {
      await deleteBackendChapter(chapter.id);
      await loadCourse(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to delete chapter.');
    } finally {
      setDeletingChapterId(null);
    }
  };

  if (loading) {
    return (
      <div className="container">
        <motion.div
          className="empty-state"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <BookOpen className="empty-state-icon" />
          <h2 className="empty-state-title">Loading Course</h2>
          <p className="empty-state-description">
            Fetching the latest course data from the backend.
          </p>
        </motion.div>
      </div>
    );
  }

  if (!course) {
    return (
      <div className="container">
        <motion.div
          className="empty-state"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <AlertCircle className="empty-state-icon" />
          <h2 className="empty-state-title">Course Not Found</h2>
          <p className="empty-state-description">
            The course you&apos;re looking for doesn&apos;t exist or has been removed.
          </p>
          <Link
            to="/courses"
            className="btn btn-primary"
            style={{ marginTop: 'var(--space-lg)', display: 'inline-flex' }}
          >
            Back to Courses
          </Link>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="container" style={{ paddingBottom: 'var(--space-3xl)' }}>
      {/* Breadcrumb */}
      <motion.nav
        className="breadcrumb"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.05 }}
        style={{ paddingTop: 'var(--space-lg)' }}
      >
        <Link to="/">Home</Link>
        <ChevronRight size={14} className="breadcrumb-separator" />
        <Link to="/courses">Courses</Link>
        <ChevronRight size={14} className="breadcrumb-separator" />
        <span>{course.title}</span>
      </motion.nav>

      {/* Course Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass"
        style={{
          borderRadius: 'var(--radius-xl)',
          padding: 'var(--space-2xl)',
          marginBottom: 'var(--space-2xl)',
        }}
      >
        <div
          className="grid grid-cols-1 md:grid-cols-[280px_1fr]"
          style={{ gap: 'var(--space-2xl)' }}
        >
          <div
            className="flex-center"
            style={{
              width: '100%',
              aspectRatio: '4/3',
              borderRadius: 'var(--radius-lg)',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-muted)',
            }}
          >
            <BookOpen size={52} />
          </div>

          {/* Info */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-md)',
            }}
          >
            <h1 className="heading-2">{course.title}</h1>
            <p className="text-body">
              {course.description || 'No description has been added for this course yet.'}
            </p>

            <div
              style={{
                display: 'flex',
                gap: 'var(--space-sm)',
                flexWrap: 'wrap',
                alignItems: 'center',
              }}
            >
              <span className="badge badge-blue">
                {chapters.length} {chapters.length === 1 ? 'chapter' : 'chapters'}
              </span>
              {course.created_at ? (
                <span className="text-small" style={{ color: 'var(--text-secondary)' }}>
                  Created {new Date(course.created_at).toLocaleDateString()}
                </span>
              ) : null}
            </div>
          </div>
        </div>
      </motion.div>

      {error ? (
        <div
          className="glass"
          style={{
            padding: 'var(--space-md)',
            marginBottom: 'var(--space-xl)',
            color: 'var(--danger, #ef4444)',
          }}
        >
          {error}
        </div>
      ) : null}

      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="glass"
        style={{
          padding: 'var(--space-xl)',
          borderRadius: 'var(--radius-xl)',
          marginBottom: 'var(--space-2xl)',
        }}
      >
        <div className="flex-between" style={{ gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          <div>
            <h2 className="heading-4" style={{ marginBottom: 'var(--space-xs)' }}>
              Manage Chapters
            </h2>
            <p className="text-small" style={{ color: 'var(--text-secondary)' }}>
              Add or remove chapters for this course.
            </p>
          </div>
          <button
            className={showChapterForm ? 'btn btn-secondary' : 'btn btn-primary'}
            onClick={() => setShowChapterForm((value) => !value)}
          >
            {showChapterForm ? <X size={16} /> : <Plus size={16} />}
            {showChapterForm ? 'Cancel' : 'Add Chapter'}
          </button>
        </div>

        {showChapterForm ? (
          <form
            onSubmit={handleCreateChapter}
            style={{
              display: 'grid',
              gap: 'var(--space-md)',
              marginTop: 'var(--space-lg)',
              maxWidth: 720,
            }}
          >
            <input
              value={newChapterTitle}
              onChange={(event) => setNewChapterTitle(event.target.value)}
              placeholder="Chapter title"
              disabled={creatingChapter}
              style={{
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-secondary)',
                color: 'var(--text-primary)',
              }}
            />
            <textarea
              value={newChapterDescription}
              onChange={(event) => setNewChapterDescription(event.target.value)}
              placeholder="Optional description"
              disabled={creatingChapter}
              rows={3}
              style={{
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-secondary)',
                color: 'var(--text-primary)',
                resize: 'vertical',
              }}
            />
            <div>
              <button className="btn btn-primary" type="submit" disabled={creatingChapter}>
                <Plus size={16} />
                {creatingChapter ? 'Creating...' : 'Create Chapter'}
              </button>
            </div>
          </form>
        ) : null}
      </motion.section>

      {/* Chapters Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div
          className="flex-between"
          style={{ marginBottom: 'var(--space-lg)' }}
        >
          <h2 className="heading-3">Chapters</h2>
          <span
            className="text-small"
            style={{ color: 'var(--text-secondary)' }}
          >
            {chapters.length} total
          </span>
        </div>

        {chapters.length === 0 ? (
          <div className="empty-state">
            <BookOpen className="empty-state-icon" />
            <h3 className="empty-state-title">No chapters yet.</h3>
            <p className="empty-state-description">
              Create expert chapters from the Expert Video Manager.
            </p>
          </div>
        ) : (
          <div className="grid-courses">
            {chapters
              .slice()
              .sort((a, b) => a.order - b.order)
              .map((chapter, index) => {
                const hasExpertVideo = Boolean(chapter.expert_video?.url);

                return (
                  <motion.div
                    key={chapter.id}
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.25 + index * 0.025 }}
                    className={`card ${hasExpertVideo ? 'card-hover' : ''}`}
                    role={hasExpertVideo ? 'button' : undefined}
                    tabIndex={hasExpertVideo ? 0 : undefined}
                    onClick={() => {
                      if (hasExpertVideo) openCompareStudio(chapter);
                    }}
                    onKeyDown={(event) => {
                      if (hasExpertVideo && (event.key === 'Enter' || event.key === ' ')) {
                        event.preventDefault();
                        openCompareStudio(chapter);
                      }
                    }}
                    style={{
                      padding: 0,
                      overflow: 'hidden',
                      cursor: hasExpertVideo ? 'pointer' : 'default',
                    }}
                  >
                    <div
                      className="flex-center"
                      style={{
                        position: 'relative',
                        width: '100%',
                        aspectRatio: '16/9',
                        background: 'var(--bg-tertiary)',
                        color: 'var(--text-muted)',
                      }}
                    >
                      {hasExpertVideo ? (
                        <>
                          <video
                            src={chapter.expert_video?.url}
                            muted
                            playsInline
                            preload="metadata"
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover',
                              display: 'block',
                            }}
                          />
                          <div
                            className="flex-center"
                            style={{
                              position: 'absolute',
                              inset: 0,
                              background: 'rgba(0,0,0,0.28)',
                            }}
                          >
                            <Play size={32} style={{ color: '#fff' }} />
                          </div>
                        </>
                      ) : (
                        <BookOpen size={40} />
                      )}
                    </div>

                    <div style={{ padding: 'var(--space-lg)' }}>
                      <div className="flex-between" style={{ gap: 'var(--space-sm)' }}>
                        <h3 className="heading-4">{chapter.title}</h3>
                        <span className="badge badge-blue">Chapter {chapter.order}</span>
                      </div>

                      <div style={{ marginTop: 'var(--space-md)' }}>
                        <span className={`badge ${hasExpertVideo ? 'badge-green' : 'badge-yellow'}`}>
                          {hasExpertVideo ? 'Expert video linked' : 'No expert video'}
                        </span>
                      </div>

                      <div
                        style={{
                          display: 'flex',
                          gap: 'var(--space-sm)',
                          flexWrap: 'wrap',
                          marginTop: 'var(--space-md)',
                        }}
                      >
                        {hasExpertVideo ? (
                          <button
                            className="btn btn-primary"
                            onClick={(event) => {
                              event.stopPropagation();
                              openCompareStudio(chapter);
                            }}
                          >
                            <Play size={16} />
                            Watch
                          </button>
                        ) : null}
                        <button
                          className="btn btn-secondary"
                          disabled={deletingChapterId === chapter.id}
                          onClick={(event) => {
                            event.stopPropagation();
                            void handleDeleteChapter(chapter);
                          }}
                        >
                          <Trash2 size={16} />
                          {deletingChapterId === chapter.id ? 'Deleting...' : 'Delete'}
                        </button>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
          </div>
        )}
      </motion.section>
    </div>
  );
}
