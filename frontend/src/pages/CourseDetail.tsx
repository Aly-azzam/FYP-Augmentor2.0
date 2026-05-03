import { useEffect, useMemo, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Play,
  Clock,
  User,
  ChevronRight,
  AlertCircle,
} from 'lucide-react';
import { useCourseStore, useUIStore } from '../store';
import { courses, getClipsForCourse } from '../services/mock/courses';
import { Progress } from '../components/ui/progress';
import { formatDuration } from '../utils/helpers';

type BackendChapterClip = {
  id: string;
  title: string;
  duration: number;
  description: string;
  thumbnail: string;
  keyPoints: string[];
  expertVideoUrl?: string;
};

type BackendChapter = {
  id: string;
  course_id: string;
  title: string;
  order: number;
  expert_video?: {
    url: string;
    file_path: string;
  } | null;
};

type BackendCourse = {
  id: string;
  title: string;
};

const backendCourseTitleByMockId: Record<string, string> = {
  'cut-a-straight-line': 'Cut a straight line',
};

export default function CourseDetail() {
  const { courseId } = useParams<{ courseId: string }>();
  const navigate = useNavigate();
  const setSelectedCourse = useCourseStore((s) => s.setSelectedCourse);
  const setSelectedClip = useCourseStore((s) => s.setSelectedClip);
  const setRobotMessage = useUIStore((s) => s.setRobotMessage);
  const [backendClips, setBackendClips] = useState<BackendChapterClip[]>([]);

  const course = useMemo(
    () => courses.find((c) => c.id === courseId),
    [courseId],
  );

  const mockClips = useMemo(() => (courseId ? getClipsForCourse(courseId) : []), [courseId]);

  const clips = useMemo(() => {
    if (
      (courseId === 'pottery-wheel' || courseId === 'cut-a-straight-line') &&
      backendClips.length > 0
    ) {
      return backendClips;
    }
    return mockClips;
  }, [backendClips, courseId, mockClips]);

  useEffect(() => {
    if (course) {
      setSelectedCourse(course.id);
      setRobotMessage(
        course.progress > 0
          ? `You're viewing "${course.title}". You've completed ${course.progress}% — keep going!`
          : `Welcome to "${course.title}"! Pick a clip below to start learning.`,
      );
    }
    return () => {
      setRobotMessage(null);
    };
  }, [course, setSelectedCourse, setRobotMessage]);

  useEffect(() => {
    let cancelled = false;

    const loadBackendChapterClips = async () => {
      if (courseId !== 'pottery-wheel' && courseId !== 'cut-a-straight-line') {
        setBackendClips([]);
        return;
      }

      try {
        let chaptersUrl = '/api/chapters';
        const backendCourseTitle = courseId ? backendCourseTitleByMockId[courseId] : undefined;
        if (backendCourseTitle) {
          const coursesResponse = await fetch('/api/courses');
          if (!coursesResponse.ok) {
            throw new Error('Failed to load backend courses');
          }

          const backendCourses = (await coursesResponse.json()) as BackendCourse[];
          const backendCourse = backendCourses.find(
            (item) => item.title.toLowerCase() === backendCourseTitle.toLowerCase(),
          );
          if (backendCourse) {
            chaptersUrl = `/api/chapters?course_id=${encodeURIComponent(backendCourse.id)}`;
          }
        }

        const response = await fetch(chaptersUrl);
        if (!response.ok) {
          throw new Error('Failed to load chapters');
        }

        const payload = (await response.json()) as BackendChapter[];
        const chapterClips: BackendChapterClip[] = payload.map((chapter) => ({
          id: chapter.id,
          title: chapter.title,
          duration: courseId === 'cut-a-straight-line' ? 15 : 10,
          description: chapter.expert_video
            ? 'This chapter has a linked expert video from backend storage.'
            : 'No expert video linked yet. Upload one from Expert Upload.',
          thumbnail: course?.thumbnail ?? '/course-pottery.jpg',
          expertVideoUrl: chapter.expert_video?.url,
          keyPoints: chapter.expert_video
            ? ['Chapter is linked to a real expert video', 'Open Compare Studio to use this exact expert reference']
            : ['Upload an expert video for this chapter', 'Then return here and open Compare Studio'],
        }));

        chapterClips.sort((a, b) => a.title.localeCompare(b.title));
        if (!cancelled) {
          setBackendClips(chapterClips);
        }
      } catch {
        if (!cancelled) {
          setBackendClips([]);
        }
      }
    };

    void loadBackendChapterClips();

    return () => {
      cancelled = true;
    };
  }, [course?.thumbnail, courseId]);

  const handleClipClick = (clipId: string) => {
    if (!course) return;

    setSelectedCourse(course.id);
    setSelectedClip(clipId);
    navigate(`/compare?courseId=${encodeURIComponent(course.id)}&clipId=${encodeURIComponent(clipId)}`);
  };

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
          {/* Thumbnail */}
          <div
            style={{
              width: '100%',
              aspectRatio: '4/3',
              borderRadius: 'var(--radius-lg)',
              background: 'var(--bg-tertiary)',
              backgroundImage: `url(${course.thumbnail})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
            }}
          />

          {/* Info */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-md)',
            }}
          >
            <h1 className="heading-2">{course.title}</h1>
            <p className="text-body">{course.description}</p>

            <div
              style={{
                display: 'flex',
                gap: 'var(--space-sm)',
                flexWrap: 'wrap',
                alignItems: 'center',
              }}
            >
              <span
                className={`difficulty-badge difficulty-${course.difficulty}`}
              >
                {course.difficulty}
              </span>
              <span className="badge badge-blue">{course.category}</span>
              <span
                className="text-small"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  color: 'var(--text-secondary)',
                }}
              >
                <User size={14} /> {course.instructor}
              </span>
              <span
                className="text-small"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  color: 'var(--text-secondary)',
                }}
              >
                <Clock size={14} /> {course.estimatedTime}
              </span>
            </div>

            {/* Progress */}
            <div>
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
              <Progress value={course.progress} />
            </div>

            {/* CTA */}
            <div>
              <button
                className="btn btn-primary"
                onClick={() => {
                  if (clips.length > 0) handleClipClick(clips[0].id);
                }}
              >
                <Play size={16} />
                {course.progress > 0 ? 'Continue Learning' : 'Start Course'}
              </button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Clips Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div
          className="flex-between"
          style={{ marginBottom: 'var(--space-lg)' }}
        >
          <h2 className="heading-3">Course Clips</h2>
          <span
            className="text-small"
            style={{ color: 'var(--text-secondary)' }}
          >
            {clips.length} videos
          </span>
        </div>

        <div
          className="horizontal-scroll"
          style={{ paddingBottom: 'var(--space-md)' }}
        >
          {clips.map((clip, index) => (
            <motion.div
              key={clip.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 + index * 0.025 }}
              className="card card-hover"
              style={{ width: 200, cursor: 'pointer', overflow: 'hidden' }}
              onClick={() => handleClipClick(clip.id)}
            >
              {/* Thumbnail with play overlay */}
              <div
                style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '16/9',
                  background: 'var(--bg-tertiary)',
                  overflow: 'hidden',
                }}
              >
                {clip.expertVideoUrl ? (
                  <video
                    src={clip.expertVideoUrl}
                    muted
                    playsInline
                    preload="metadata"
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                  />
                ) : (
                  <div
                    style={{
                      width: '100%',
                      height: '100%',
                      backgroundImage: `url(${clip.thumbnail})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                    }}
                  />
                )}

                <div
                  className="flex-center"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'rgba(0,0,0,0.4)',
                    opacity: 0,
                    transition: 'opacity var(--transition-fast)',
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.opacity = '1';
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.opacity = '0';
                  }}
                >
                  <Play size={28} style={{ color: '#fff' }} />
                </div>

                {clip.expertVideoUrl && (
                  <span
                    className="badge badge-blue"
                    style={{ position: 'absolute', top: 6, left: 6 }}
                  >
                    Real Expert Video
                  </span>
                )}

                <span
                  style={{
                    position: 'absolute',
                    bottom: 4,
                    right: 4,
                    background: 'rgba(0,0,0,0.75)',
                    color: '#fff',
                    padding: '2px 6px',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.7rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  {formatDuration(clip.duration)}
                </span>
              </div>

              {/* Title */}
              <div style={{ padding: 'var(--space-sm)' }}>
                <p
                  className="text-small"
                  style={{
                    fontWeight: 500,
                    color: 'var(--text-primary)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {clip.title}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>
    </div>
  );
}
