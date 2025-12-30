import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Camera, Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const Navbar = () => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[90%] max-w-5xl">
      <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border border-gray-200 dark:border-gray-700 rounded-2xl shadow-lg px-8 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <Camera className="w-6 h-6 text-blue-600 dark:text-blue-400 stroke-2" strokeWidth={1.5} />
            <span className="text-xl font-semibold text-gray-900 dark:text-white">
              E-Raksha
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className={`text-sm font-medium transition-colors ${
                isActive('/')
                  ? 'text-blue-600 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Home
            </Link>
            <Link
              to="/workbench"
              className={`text-sm font-medium transition-colors ${
                isActive('/workbench')
                  ? 'text-blue-600 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Analysis
            </Link>
            <Link
              to="/dashboard"
              className={`text-sm font-medium transition-colors ${
                isActive('/dashboard')
                  ? 'text-blue-600 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Dashboard
            </Link>
            <Link
              to="/analytics"
              className={`text-sm font-medium transition-colors ${
                isActive('/analytics')
                  ? 'text-blue-600 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Statistics
            </Link>
            <Link
              to="/contact"
              className={`text-sm font-medium transition-colors ${
                isActive('/contact')
                  ? 'text-blue-600 dark:text-blue-400'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              Contact
            </Link>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              ) : (
                <Sun className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;