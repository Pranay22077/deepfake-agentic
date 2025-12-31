import React from 'react';
import { useTheme } from '../context/ThemeContext';
import LiquidEther from './LiquidEther';

export default function LiquidEtherBackground() {
  const { theme } = useTheme();
  
  // Theme-aware colors
  const colors = theme === 'light'
    ? ['#3B82F6', '#A78BFA', '#60A5FA'] // Light mode: blue shades
    : ['#5227FF', '#FF9FFC', '#B19EEF']; // Dark mode: purple/pink shades
  
  return (
    <div className="fixed inset-0 pointer-events-none opacity-20 dark:opacity-30" style={{ zIndex: 1 }}>
      <LiquidEther
        colors={colors}
        mouseForce={15}
        cursorSize={80}
        isViscous={false}
        viscous={30}
        iterationsViscous={32}
        iterationsPoisson={32}
        resolution={0.4}
        isBounce={false}
        autoDemo={true}
        autoSpeed={0.3}
        autoIntensity={1.8}
        takeoverDuration={0.25}
        autoResumeDelay={3000}
        autoRampDuration={0.6}
      />
    </div>
  );
}
