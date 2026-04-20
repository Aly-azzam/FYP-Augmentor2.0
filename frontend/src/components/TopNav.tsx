import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BookOpen,
  BarChart3,
  RefreshCw,
  Clock,
  Trophy,
  Upload,
  Sun,
  Moon,
  Bot,
  User,
  Settings,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from './ui/dropdown-menu';
import { useThemeStore, useUIStore } from '../store';

const navLinks = [
  { label: 'Courses', path: '/courses', icon: BookOpen },
  { label: 'My Learning', path: '/progress', icon: BarChart3 },
  { label: 'Compare Studio', path: '/compare', icon: RefreshCw },
  { label: 'History', path: '/history', icon: Clock },
  { label: 'Achievements', path: '/achievements', icon: Trophy },
  { label: 'Expert Upload', path: '/expert-videos', icon: Upload },
] as const;

export default function TopNav() {
  const location = useLocation();
  const { isDark, toggleTheme } = useThemeStore();
  const { showRobot, setShowRobot } = useUIStore();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 glass"
      style={{ height: 64, backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)' }}
    >
      <div className="container h-full flex-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-0 text-xl no-underline">
          <span className="font-normal text-[var(--text-primary)]">Aug</span>
          <span className="font-bold gradient-text">Mentor 2.0</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-1">
          {navLinks.map(({ label, path, icon: Icon }) => (
            <Link
              key={path}
              to={path}
              className={`nav-link flex items-center gap-1.5 px-3 py-2 ${isActive(path) ? 'active' : ''}`}
            >
              <Icon size={16} />
              <span>{label}</span>
            </Link>
          ))}
        </nav>

        {/* Right controls */}
        <div className="flex items-center gap-2">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="btn btn-ghost p-2 rounded-lg"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          {/* Robot toggle */}
          <button
            onClick={() => setShowRobot(!showRobot)}
            className={`btn p-2 rounded-lg transition-colors ${
              showRobot
                ? 'bg-[var(--accent-soft)] text-[var(--accent-primary)]'
                : 'btn-ghost'
            }`}
            aria-label="Toggle robot assistant"
          >
            <Bot size={18} />
          </button>

          {/* User dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-2 btn btn-ghost p-1.5 rounded-full">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[var(--accent-primary)] to-[var(--blue-400)] flex items-center justify-center text-white text-sm font-semibold">
                  AJ
                </div>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-[var(--text-primary)]">
                    Alex Johnson
                  </span>
                  <span className="text-xs text-[var(--text-muted)]">alex@augmentor.dev</span>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link to="/profile" className="flex items-center gap-2 no-underline">
                  <User size={14} />
                  Profile
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link to="/settings" className="flex items-center gap-2 no-underline">
                  <Settings size={14} />
                  Settings
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={toggleTheme}>
                {isDark ? <Sun size={14} /> : <Moon size={14} />}
                {isDark ? 'Light mode' : 'Dark mode'}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-[var(--error)]">
                <LogOut size={14} />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Mobile hamburger */}
          <button
            className="btn btn-ghost p-2 md:hidden"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile slide-out panel */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, x: '100%' }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed inset-0 top-16 z-40 md:hidden"
            style={{ background: 'var(--bg-secondary)' }}
          >
            <nav className="flex flex-col p-4 gap-1">
              {navLinks.map(({ label, path, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  onClick={() => setMobileOpen(false)}
                  className={`nav-link flex items-center gap-3 px-4 py-3 rounded-[var(--radius-md)] no-underline ${
                    isActive(path)
                      ? 'active bg-[var(--accent-soft)]'
                      : 'hover:bg-[var(--bg-tertiary)]'
                  }`}
                >
                  <Icon size={18} />
                  <span className="text-base">{label}</span>
                </Link>
              ))}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
