import { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'sonner';
import { useThemeStore } from './store';
import TopNav from './components/TopNav';
import RobotAssistant from './components/RobotAssistant';
import Dashboard from './pages/Home';
import CourseLibrary from './pages/Courses';
import CourseDetail from './pages/CourseDetail';
import CompareStudio from './pages/CompareStudio';
import ProgressPage from './pages/ProgressPage';
import HistoryPage from './pages/HistoryPage';
import AchievementsPage from './pages/AchievementsPage';
import SettingsPage from './pages/SettingsPage';
import ProfilePage from './pages/ProfilePage';
import ExpertVideoManager from './pages/ExpertVideoManager';
import NotFound from './pages/NotFound';

export default function App() {
  const isDark = useThemeStore((s) => s.isDark);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.add('light');
    }
  }, [isDark]);

  return (
    <BrowserRouter>
      <div className={`app ${isDark ? 'dark' : 'light'}`}>
        <TopNav />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/courses" element={<CourseLibrary />} />
            <Route path="/courses/:courseId" element={<CourseDetail />} />
            <Route path="/compare" element={<CompareStudio />} />
            <Route path="/progress" element={<ProgressPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/achievements" element={<AchievementsPage />} />
            <Route path="/expert-videos" element={<ExpertVideoManager />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <RobotAssistant />
        <Toaster
          position="bottom-right"
          theme={isDark ? 'dark' : 'light'}
          richColors
          closeButton
        />
      </div>
    </BrowserRouter>
  );
}
