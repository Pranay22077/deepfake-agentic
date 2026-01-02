import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { getAnalyticsStats } from '../../lib/supabase';

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    fakeDetected: 0,
    realDetected: 0,
    recentAnalyses: 0,
    averageConfidence: 0
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getAnalyticsStats();
        setStats(data);
      } catch (error) {
        console.error('Error fetching stats:', error);
      }
    };

    fetchStats();
  }, []);

  const faqs = [
    {
      q: 'What is Interceptor?',
      a: `Interceptor is an agentic AI system for deepfake detection developed for the E-Raksha Hackathon 2026. It uses 6 specialist neural networks (BG, AV, CM, RR, LL, TM models) coordinated by a LangGraph agent to provide comprehensive video authenticity analysis with ${(stats.averageConfidence * 100).toFixed(1)}% detection confidence.`,
    },
    {
      q: 'How is Interceptor different from other deepfake detectors?',
      a: 'Unlike single-model detectors, Interceptor uses an agentic approach with LangGraph. An intelligent routing engine analyzes video characteristics (compression, lighting, temporal patterns) and routes to specialized models. This multi-model ensemble provides more reliable detection across diverse real-world scenarios.',
    },
    {
      q: 'What are the 6 specialist models?',
      a: 'BG-Model (Baseline Generalist - 86.25% accuracy), AV-Model (Audio-Visual - 93.0%), CM-Model (Compression - 80.83%), RR-Model (Re-recording - 85.0%), LL-Model (Low-light - 93.42%), and TM-Model (Temporal - 78.5%). Each is trained for specific video conditions.',
    },
    {
      q: 'Does this work on low-quality or compressed videos?',
      a: 'Yes. The CM-Model (Compression Specialist) is specifically trained on videos compressed at 200-800 kbps, common on WhatsApp, Instagram, and YouTube. The RR-Model handles screen-recorded content with moiré patterns.',
    },
    {
      q: 'How fast is the analysis?',
      a: 'Average processing time is 2.1 seconds. The agentic routing can save up to 83% computation by accepting high-confidence baseline predictions without invoking all specialists.',
    },
    {
      q: 'Is my video stored or shared?',
      a: 'No. Uploaded videos are processed securely and deleted after analysis. Videos are never used for training without explicit user consent and human verification.',
    },
    {
      q: 'How accurate is Interceptor?',
      a: `Overall detection confidence is ${(stats.averageConfidence * 100).toFixed(1)}%. Individual model accuracies range from 78.5% (TM-Model) to 93.42% (LL-Model). The system prioritizes reliability over overconfidence - uncertain videos are flagged rather than forced predictions.`,
    },
    {
      q: 'What happens when confidence is low?',
      a: 'The agent uses confidence-based routing: High (≥85%) accepts baseline, Medium (65-85%) routes to 2-3 relevant specialists, Low (<65%) invokes all 6 specialists. Uncertain results are flagged for human review.',
    },
    {
      q: 'Does Interceptor analyze audio?',
      a: 'Yes. The AV-Model (Audio-Visual) analyzes lip-sync consistency and audio-visual correlation. Audio is extracted and processed alongside video frames for comprehensive analysis.',
    },
    {
      q: 'Can this be used offline?',
      a: 'Yes. All models are optimized for edge deployment with INT8 quantization. Total model size is 47.2M parameters (~512MB memory). Offline mobile deployment is supported.',
    },
    {
      q: 'What video formats are supported?',
      a: 'MP4, AVI, MOV, MKV, and WebM formats up to 100MB. Videos are decoded, frames sampled, faces detected using MTCNN, and processed through the agentic pipeline.',
    },
    {
      q: 'How does the explainability work?',
      a: 'Interceptor provides Grad-CAM heatmaps highlighting suspicious regions, confidence breakdowns per model, and human-readable explanations of why a video was flagged.',
    },
    {
      q: 'Who is Interceptor designed for?',
      a: 'Law enforcement and investigators, journalists and fact-checkers, digital forensics teams, field operatives (offline edge deployment), and general users concerned about misinformation.',
    },
    {
      q: 'Can Interceptor be integrated via API?',
      a: 'Yes. FastAPI backend provides RESTful endpoints: POST /predict for analysis, GET /stats for system statistics, GET /health for status checks. Full Swagger documentation available.',
    },
    {
      q: 'What is the technology stack?',
      a: 'PyTorch for neural networks, LangGraph for agentic orchestration, FastAPI for backend, React for frontend, MTCNN for face detection, and Docker for deployment. ResNet18-based architectures with knowledge distillation.',
    },
    {
      q: 'Who built Interceptor?',
      a: 'Interceptor was developed by a multidisciplinary team for the E-Raksha Hackathon 2026 (Problem Statement II) - National Cyber Challenge by eDC IIT Delhi in collaboration with CyberPeace.',
    },
  ];

  const toggleQuestion = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <div className="mb-12">
          <h1 className="text-3xl md:text-4xl mb-3 font-bold text-gray-900 dark:text-white">
            Frequently Asked Questions
          </h1>
          <p className="text-base text-gray-600 dark:text-gray-400">
            Find answers to common questions about Interceptor
          </p>
        </div>

        {/* FAQ Accordion */}
        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md overflow-hidden"
            >
              <button
                onClick={() => toggleQuestion(index)}
                className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
              >
                <div className="flex items-start gap-3 flex-1">
                  <span className="text-blue-600 dark:text-blue-400 font-semibold">
                    Q:
                  </span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {faq.q}
                  </span>
                </div>
                <div className="flex-shrink-0 ml-4">
                  {openIndex === index ? (
                    <ChevronUp className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  )}
                </div>
              </button>
              <div
                className={`overflow-hidden transition-all duration-700 ease-in-out ${
                  openIndex === index ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                }`}
              >
                <div className="px-6 pb-4 border-t border-gray-200 dark:border-gray-800 pt-4 mt-2">
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    {faq.a}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Contact CTA */}
        <div className="mt-12 rounded-xl p-8 text-center bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800">
          <h3 className="text-xl mb-3 font-bold text-gray-900 dark:text-white">
            Still have questions?
          </h3>
          <p className="text-base mb-6 text-gray-600 dark:text-gray-400">
            Our team is here to help you get started with Interceptor
          </p>
          <a
            href="/contact"
            className="inline-flex items-center px-6 py-3 rounded-lg transition-colors bg-blue-600 hover:bg-blue-700 text-white font-medium"
          >
            Contact Us
          </a>
        </div>
      </div>
    </div>
  );
};

export default FAQ;