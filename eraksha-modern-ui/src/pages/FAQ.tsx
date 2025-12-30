import React from 'react';

const FAQ = () => {
  const faqs = [
    {
      question: 'How accurate is the deepfake detection?',
      answer: 'Our agentic system achieves 94.9% confidence through multiple specialist models working together. The system uses advanced AI techniques to analyze various aspects of video content.',
    },
    {
      question: 'What video formats are supported?',
      answer: 'We support MP4, AVI, MOV, and WebM formats. Maximum file size is 100MB for optimal processing speed.',
    },
    {
      question: 'How long does analysis take?',
      answer: 'Most videos are processed within 2-4 seconds, depending on file size and complexity. Our system is optimized for speed without compromising accuracy.',
    },
    {
      question: 'Is my data secure?',
      answer: 'Yes, we take privacy seriously. Videos are processed locally and not stored on our servers. All analysis happens in real-time without data retention.',
    },
    {
      question: 'Can I use this for commercial purposes?',
      answer: 'Please contact us for commercial licensing options. We offer enterprise solutions with additional features and support.',
    },
  ];

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Frequently Asked Questions
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Find answers to common questions about our deepfake detection system.
          </p>
        </div>

        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8"
            >
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                {faq.question}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                {faq.answer}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FAQ;