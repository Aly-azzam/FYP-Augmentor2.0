import type { Course, VideoClip } from '@/types';

export type BackendCourse = {
  id: string;
  title: string;
  description?: string | null;
  created_at?: string;
  updated_at?: string;
  thumbnail_url?: string | null;
};

export type BackendChapter = {
  id: string;
  course_id: string;
  title: string;
  order: number;
  expert_video?: {
    id?: string;
    file_path?: string;
    url?: string;
    duration_seconds?: number | null;
  } | null;
};

export type BackendCourseDetail = BackendCourse & {
  chapters?: {
    id: string;
    title: string;
    order: number;
    has_expert_video: boolean;
  }[];
};

const DEFAULT_THUMBNAIL = '/course-leather.jpg';
const DEFAULT_INSTRUCTOR = 'AugMentor Team';
const DEFAULT_CATEGORY = 'Practice';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Request failed: ${url}`);
  }
  return (await response.json()) as T;
}

export async function fetchBackendCourses(): Promise<BackendCourse[]> {
  return fetchJson<BackendCourse[]>('/api/courses');
}

export async function fetchBackendChapters(courseId?: string): Promise<BackendChapter[]> {
  const query = courseId ? `?course_id=${encodeURIComponent(courseId)}` : '';
  return fetchJson<BackendChapter[]>(`/api/chapters${query}`);
}

export async function fetchBackendCourseDetail(courseId: string): Promise<BackendCourseDetail> {
  return fetchJson<BackendCourseDetail>(`/api/courses/${encodeURIComponent(courseId)}`);
}

export async function createBackendChapter(payload: {
  course_id: string;
  title: string;
  description?: string | null;
}): Promise<BackendChapter> {
  return fetchJson<BackendChapter>('/api/chapters', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function deleteBackendChapter(chapterId: string): Promise<void> {
  const response = await fetch(`/api/chapters/${encodeURIComponent(chapterId)}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const errorBody = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(errorBody?.detail ?? 'Failed to delete chapter.');
  }
}

export function toCourse(course: BackendCourse, chapterCount = 0): Course {
  const safeChapterCount = Math.max(0, chapterCount);
  return {
    id: course.id,
    title: course.title,
    description: course.description || 'Practice with backend-managed expert reference videos.',
    difficulty: 'beginner',
    totalClips: safeChapterCount,
    estimatedTime: safeChapterCount > 0 ? `${safeChapterCount * 10}m` : '0m',
    progress: 0,
    thumbnail: course.thumbnail_url || DEFAULT_THUMBNAIL,
    category: DEFAULT_CATEGORY,
    instructor: DEFAULT_INSTRUCTOR,
  };
}

export async function fetchCourses(): Promise<Course[]> {
  const [backendCourses, chapters] = await Promise.all([
    fetchBackendCourses(),
    fetchBackendChapters(),
  ]);
  const chapterCounts = chapters.reduce<Record<string, number>>((counts, chapter) => {
    counts[chapter.course_id] = (counts[chapter.course_id] ?? 0) + 1;
    return counts;
  }, {});

  return backendCourses.map((course) => toCourse(course, chapterCounts[course.id] ?? 0));
}

export async function fetchCourse(courseId: string): Promise<Course> {
  const detail = await fetchBackendCourseDetail(courseId);
  return toCourse(detail, detail.chapters?.length ?? 0);
}

export async function fetchClipsForCourse(courseId: string, thumbnail = DEFAULT_THUMBNAIL): Promise<VideoClip[]> {
  const chapters = await fetchBackendChapters(courseId);
  return chapters
    .slice()
    .sort((a, b) => a.order - b.order)
    .map((chapter) => ({
      id: chapter.id,
      title: chapter.title,
      duration: Math.round(chapter.expert_video?.duration_seconds ?? 10),
      description: chapter.expert_video
        ? 'This chapter has a linked expert video from backend storage.'
        : 'No expert video linked yet. Upload one from Expert Video Manager.',
      thumbnail,
      expertVideoUrl: chapter.expert_video?.url,
      keyPoints: chapter.expert_video
        ? [
            'Chapter is linked to a real backend expert video',
            'Open Compare Studio to use this exact expert reference',
          ]
        : [
            'Upload an expert video for this chapter',
            'Then return here and open Compare Studio',
          ],
    }));
}
