import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { Upload, RefreshCw } from 'lucide-react';

type ExpertVideo = {
  id: string;
  chapter_id: string;
  file_path: string;
  url: string;
  duration_seconds?: number | null;
  fps?: number | null;
};

type Chapter = {
  id: string;
  course_id: string;
  title: string;
  order: number;
  expert_video?: ExpertVideo | null;
};

export default function ExpertVideoManager() {
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<Record<string, File | null>>({});
  const [uploadingChapterId, setUploadingChapterId] = useState<string | null>(null);

  const loadChapters = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/chapters');
      if (!response.ok) {
        throw new Error('Failed to load chapters');
      }
      const payload = (await response.json()) as Chapter[];
      setChapters(payload);
    } catch {
      setError('Could not load chapters. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadChapters();
  }, []);

  const handleFileChange = (chapterId: string, file: File | null) => {
    setSelectedFiles((prev) => ({ ...prev, [chapterId]: file }));
  };

  const handleUpload = async (chapterId: string) => {
    const file = selectedFiles[chapterId];
    if (!file) {
      toast.error('Please choose a video file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setUploadingChapterId(chapterId);
    try {
      const response = await fetch(`/api/chapters/${chapterId}/expert-video`, {
        method: 'POST',
        body: formData,
      });

      let payload: any = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        throw new Error(
          payload?.detail || payload?.message || `Upload failed with status ${response.status}`,
        );
      }

      toast.success('Expert video updated.');
      setSelectedFiles((prev) => ({ ...prev, [chapterId]: null }));
      await loadChapters();
    } catch (uploadError) {
      const message =
        uploadError instanceof Error ? uploadError.message : 'Failed to upload expert video.';
      toast.error(message);
    } finally {
      setUploadingChapterId(null);
    }
  };

  return (
    <div className="container" style={{ paddingBottom: 'var(--space-3xl)' }}>
      <div className="page-header">
        <h1 className="page-title">Expert Video Manager</h1>
        <p className="page-subtitle">
          Upload or replace the expert video linked to each chapter.
        </p>
      </div>

      <div style={{ display: 'flex', gap: 'var(--space-sm)', marginBottom: 'var(--space-lg)' }}>
        <button className="btn btn-secondary" onClick={() => void loadChapters()}>
          <RefreshCw size={16} />
          Refresh
        </button>
        <Link to="/courses" className="btn btn-ghost">
          Back to Courses
        </Link>
      </div>

      {loading ? (
        <div className="glass" style={{ padding: 'var(--space-xl)' }}>
          Loading chapters...
        </div>
      ) : error ? (
        <div className="glass" style={{ padding: 'var(--space-xl)', color: 'var(--danger, #dc2626)' }}>
          {error}
        </div>
      ) : (
        <div style={{ display: 'grid', gap: 'var(--space-lg)' }}>
          {chapters.map((chapter, index) => {
            const selectedFile = selectedFiles[chapter.id];
            const isUploading = uploadingChapterId === chapter.id;
            return (
              <motion.div
                key={chapter.id}
                className="glass"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.04 }}
                style={{ padding: 'var(--space-lg)' }}
              >
                <div className="flex-between" style={{ gap: 'var(--space-md)', flexWrap: 'wrap' }}>
                  <div>
                    <h3 className="heading-4" style={{ marginBottom: 'var(--space-xs)' }}>
                      {chapter.title}
                    </h3>
                    <p className="text-small" style={{ color: 'var(--text-secondary)' }}>
                      Chapter order: {chapter.order}
                    </p>
                  </div>
                  <span className={chapter.expert_video ? 'badge badge-blue' : 'badge'}>
                    {chapter.expert_video ? 'Has expert video' : 'No expert video'}
                  </span>
                </div>

                {chapter.expert_video ? (
                  <div style={{ marginTop: 'var(--space-md)' }}>
                    <video
                      src={chapter.expert_video.url}
                      controls
                      preload="metadata"
                      style={{
                        width: '100%',
                        maxWidth: 560,
                        aspectRatio: '16/9',
                        borderRadius: 'var(--radius-md)',
                        background: 'var(--bg-tertiary)',
                      }}
                    />
                    <p className="text-small" style={{ color: 'var(--text-muted)', marginTop: 'var(--space-xs)' }}>
                      Current path: {chapter.expert_video.file_path}
                    </p>
                  </div>
                ) : null}

                <div
                  style={{
                    display: 'flex',
                    gap: 'var(--space-sm)',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    marginTop: 'var(--space-md)',
                  }}
                >
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.webm"
                    onChange={(e) => handleFileChange(chapter.id, e.target.files?.[0] ?? null)}
                  />
                  <button
                    className="btn btn-primary"
                    onClick={() => void handleUpload(chapter.id)}
                    disabled={isUploading}
                  >
                    <Upload size={14} />
                    {isUploading ? 'Uploading...' : 'Upload Expert Video'}
                  </button>
                  {selectedFile ? (
                    <span className="text-small" style={{ color: 'var(--text-secondary)' }}>
                      Selected: {selectedFile.name}
                    </span>
                  ) : null}
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
