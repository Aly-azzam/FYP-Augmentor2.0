import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Home, SearchX } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="container section">
      <motion.div
        className="empty-state"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <SearchX className="empty-state-icon" style={{ width: 80, height: 80 }} />
        <h1 className="empty-state-title" style={{ fontSize: '2rem' }}>
          Page Not Found
        </h1>
        <p className="empty-state-description" style={{ marginBottom: 'var(--space-xl)' }}>
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link to="/" className="btn btn-primary" style={{ gap: '0.5rem' }}>
          <Home size={16} />
          Back to Home
        </Link>
      </motion.div>
    </div>
  );
}
