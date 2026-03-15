import { useState, useRef, useMemo, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, X, MessageCircle } from 'lucide-react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float } from '@react-three/drei';
import * as THREE from 'three';
import { useUIStore } from '../store';
import type { RobotTip } from '../types';

const robotTips: RobotTip[] = [
  { id: '1', message: 'Try recording yourself from multiple angles for better analysis!', context: 'general', priority: 'medium' },
  { id: '2', message: 'Your trajectory accuracy improved 12% this week. Keep it up!', context: 'progress', priority: 'high' },
  { id: '3', message: 'Use the Compare Studio to see side-by-side differences with the reference.', context: 'compare', priority: 'medium' },
  { id: '4', message: 'Slow, controlled movements lead to better scores. Focus on precision first.', context: 'technique', priority: 'high' },
  { id: '5', message: 'You\'re on a 5-day streak! Consistency is key to mastery.', context: 'motivation', priority: 'low' },
];

function isWebGLAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
  } catch {
    return false;
  }
}

function RobotModel({ hovered, clicked }: { hovered: boolean; clicked: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const timeRef = useRef(0);
  const clickTimeRef = useRef(0);

  const eyeMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#60A5FA', emissive: '#3B82F6', emissiveIntensity: 2 }), []);
  const bodyMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#2563EB', roughness: 0.3, metalness: 0.6 }), []);
  const darkBlueMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#1E3A8A', emissive: '#1E40AF', emissiveIntensity: 0.3 }), []);
  const whiteMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#F0F4FF', roughness: 0.2, metalness: 0.1 }), []);
  const visorMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#0F172A', roughness: 0.1, metalness: 0.8 }), []);
  const antennaTipMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#F472B6', emissive: '#EC4899', emissiveIntensity: 1.5 }), []);
  const flameMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: '#F59E0B', emissive: '#F97316', emissiveIntensity: 2, transparent: true, opacity: 0.8 }), []);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    timeRef.current += delta;

    const idleBob = Math.sin(timeRef.current * 2) * 0.03;
    let yOffset = idleBob;

    if (clicked) {
      clickTimeRef.current += delta * 8;
      yOffset += Math.abs(Math.sin(clickTimeRef.current)) * 0.15;
      if (clickTimeRef.current > Math.PI) clickTimeRef.current = 0;
    }

    groupRef.current.position.y = yOffset;

    if (hovered) {
      groupRef.current.rotation.y += delta * 1.5;
    } else {
      groupRef.current.rotation.y += (0 - groupRef.current.rotation.y) * 0.05;
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.3}>
      <group ref={groupRef}>
        {/* Body - capsule shape */}
        <mesh material={bodyMaterial} position={[0, -0.15, 0]}>
          <capsuleGeometry args={[0.28, 0.3, 16, 16]} />
        </mesh>

        {/* Chest plate */}
        <mesh material={darkBlueMaterial} position={[0, -0.05, 0.22]}>
          <boxGeometry args={[0.32, 0.22, 0.06]} />
        </mesh>

        {/* Head */}
        <mesh material={whiteMaterial} position={[0, 0.42, 0]}>
          <sphereGeometry args={[0.24, 24, 24]} />
        </mesh>

        {/* Visor */}
        <mesh material={visorMaterial} position={[0, 0.42, 0.15]}>
          <boxGeometry args={[0.36, 0.16, 0.12]} />
        </mesh>

        {/* Left eye */}
        <mesh material={eyeMaterial} position={[-0.08, 0.43, 0.22]}>
          <sphereGeometry args={[0.04, 12, 12]} />
        </mesh>

        {/* Right eye */}
        <mesh material={eyeMaterial} position={[0.08, 0.43, 0.22]}>
          <sphereGeometry args={[0.04, 12, 12]} />
        </mesh>

        {/* Smile (torus arc) */}
        <mesh material={eyeMaterial} position={[0, 0.37, 0.22]} rotation={[0, 0, 0]}>
          <torusGeometry args={[0.06, 0.012, 8, 16, Math.PI]} />
        </mesh>

        {/* Antenna stalk */}
        <mesh material={bodyMaterial} position={[0, 0.7, 0]}>
          <cylinderGeometry args={[0.015, 0.015, 0.12, 8]} />
        </mesh>

        {/* Antenna tip - pink glow */}
        <mesh material={antennaTipMaterial} position={[0, 0.78, 0]}>
          <sphereGeometry args={[0.04, 12, 12]} />
        </mesh>

        {/* Left arm */}
        <mesh material={bodyMaterial} position={[-0.4, -0.1, 0]} rotation={[0, 0, 0.3]}>
          <capsuleGeometry args={[0.06, 0.22, 8, 8]} />
        </mesh>

        {/* Right arm */}
        <mesh material={bodyMaterial} position={[0.4, -0.1, 0]} rotation={[0, 0, -0.3]}>
          <capsuleGeometry args={[0.06, 0.22, 8, 8]} />
        </mesh>

        {/* Left hand */}
        <mesh material={eyeMaterial} position={[-0.48, -0.28, 0]}>
          <sphereGeometry args={[0.055, 10, 10]} />
        </mesh>

        {/* Right hand */}
        <mesh material={eyeMaterial} position={[0.48, -0.28, 0]}>
          <sphereGeometry args={[0.055, 10, 10]} />
        </mesh>

        {/* Jet flame left */}
        <mesh material={flameMaterial} position={[-0.1, -0.55, 0]} scale={[1, 1 + Math.random() * 0.3, 1]}>
          <coneGeometry args={[0.04, 0.14, 8]} />
        </mesh>

        {/* Jet flame right */}
        <mesh material={flameMaterial} position={[0.1, -0.55, 0]} scale={[1, 1 + Math.random() * 0.3, 1]}>
          <coneGeometry args={[0.04, 0.14, 8]} />
        </mesh>

        {/* Point light for eye glow */}
        <pointLight position={[0, 0.43, 0.3]} color="#3B82F6" intensity={0.4} distance={1} />
        <pointLight position={[0, 0.78, 0]} color="#EC4899" intensity={0.3} distance={0.5} />
      </group>
    </Float>
  );
}

function RobotCanvas() {
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);

  return (
    <Canvas
      dpr={[1, 1.5]}
      camera={{ fov: 40, position: [0, 0, 2.5] }}
      style={{ width: '100%', height: '100%' }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onPointerDown={() => setClicked(true)}
      onPointerUp={() => setClicked(false)}
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[3, 3, 5]} intensity={1} />
      <directionalLight position={[-2, 2, -3]} intensity={0.3} color="#93C5FD" />
      <Suspense fallback={null}>
        <RobotModel hovered={hovered} clicked={clicked} />
      </Suspense>
    </Canvas>
  );
}

export default function RobotAssistant() {
  const { showRobot, robotMessage } = useUIStore();
  const [expanded, setExpanded] = useState(false);
  const [tipIndex, setTipIndex] = useState(0);
  const webgl = useMemo(() => isWebGLAvailable(), []);

  const currentMessage = robotMessage || robotTips[tipIndex].message;

  const cycleTip = () => {
    if (!robotMessage) {
      setTipIndex((prev) => (prev + 1) % robotTips.length);
    }
  };

  if (!showRobot) return null;

  return (
    <div className="fixed z-40" style={{ bottom: 24, right: 24 }}>
      <AnimatePresence mode="wait">
        {expanded ? (
          <motion.div
            key="panel"
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            className="glass rounded-[var(--radius-xl)] overflow-hidden"
            style={{ width: 320, border: '1px solid var(--border-default)' }}
          >
            {/* Header */}
            <div className="flex-between px-4 py-3" style={{ borderBottom: '1px solid var(--border-default)' }}>
              <div className="flex items-center gap-2">
                <div
                  className="w-8 h-8 rounded-full flex-center"
                  style={{ background: 'linear-gradient(135deg, var(--accent-primary), var(--blue-400))' }}
                >
                  <Bot size={16} className="text-white" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-[var(--text-primary)]">AugBot</div>
                  <div className="text-xs text-[var(--text-muted)]">Your AI assistant</div>
                </div>
              </div>
              <button
                onClick={() => setExpanded(false)}
                className="btn btn-ghost p-1.5 rounded-lg"
              >
                <X size={16} />
              </button>
            </div>

            {/* Robot viewport */}
            <div className="relative" style={{ height: 180, background: 'var(--bg-primary)' }}>
              {webgl ? (
                <RobotCanvas />
              ) : (
                <div className="flex-center h-full">
                  <Bot size={64} className="text-[var(--accent-primary)] opacity-50" />
                </div>
              )}
            </div>

            {/* Message bubble */}
            <div className="p-4">
              <div
                className="rounded-[var(--radius-lg)] p-3 text-sm text-[var(--text-secondary)] leading-relaxed cursor-pointer"
                style={{ background: 'var(--bg-tertiary)' }}
                onClick={cycleTip}
              >
                <div className="flex items-start gap-2">
                  <MessageCircle size={14} className="text-[var(--accent-primary)] mt-0.5 shrink-0" />
                  <span>{currentMessage}</span>
                </div>
              </div>
              {!robotMessage && (
                <div className="text-xs text-[var(--text-muted)] mt-2 text-center">
                  Tap message to see more tips
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.button
            key="button"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            transition={{ type: 'spring', damping: 15, stiffness: 300 }}
            onClick={() => setExpanded(true)}
            className="relative rounded-full shadow-[var(--shadow-glow-strong)] overflow-hidden"
            style={{
              width: 56,
              height: 56,
              background: 'linear-gradient(135deg, var(--accent-primary), var(--blue-400))',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            {webgl ? (
              <RobotCanvas />
            ) : (
              <div className="flex-center w-full h-full">
                <Bot size={24} className="text-white" />
              </div>
            )}
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
}
