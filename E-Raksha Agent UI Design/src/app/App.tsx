import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import AnalysisWorkbench from './pages/AnalysisWorkbench';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Contact from './pages/Contact';
import FAQ from './pages/FAQ';

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-purple-100 via-blue-100 to-cyan-100 dark:from-gray-950 dark:via-blue-950 dark:to-purple-950 transition-colors relative overflow-hidden">
          {/* Animated gradient background */}
          <div className="fixed inset-0 opacity-30 dark:opacity-20">
            <div className="absolute top-0 left-0 w-96 h-96 bg-purple-400 dark:bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob"></div>
            <div className="absolute top-0 right-0 w-96 h-96 bg-blue-400 dark:bg-blue-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000"></div>
            <div className="absolute bottom-0 left-1/2 w-96 h-96 bg-cyan-400 dark:bg-cyan-600 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000"></div>
          </div>
          
          <div className="relative z-10">
            <Navbar />
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/workbench" element={<AnalysisWorkbench />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/faq" element={<FAQ />} />
            </Routes>
          </div>
        </div>
      </Router>
    </ThemeProvider>
  );
}
