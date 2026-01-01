import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const faqs = [
    {
      q: 'What does this system do?',
      a: 'This system analyzes videos to determine whether they are real or AI-generated (deepfake). It evaluates visual and temporal cues using multiple AI models and provides a confidence score along with an explanation of its decision.',
    },
    {
      q: 'How is this different from other deepfake detectors?',
      a: 'Unlike single-model detectors, our system uses an agent-based approach. An intelligent controller decides which specialized AI model to use depending on the video\'s quality, compression level, and artifacts, making detection more reliable in real-world scenarios.',
    },
    {
      q: 'Does this work on low-quality or compressed videos?',
      a: 'Yes. The system includes specialist models trained specifically for compressed, re-encoded, and screen-recorded videos, which are common on platforms like WhatsApp, Instagram, and YouTube.',
    },
    {
      q: 'Is this detection done in real time?',
      a: 'The system provides near real-time analysis for short videos. Processing time may vary depending on video length, resolution, and complexity.',
    },
    {
      q: 'Is my video stored or shared?',
      a: 'No. Uploaded videos are not publicly shared. Videos are processed securely and can be automatically deleted after analysis unless the user explicitly consents to contribute data for research and model improvement.',
    },
    {
      q: 'Does the system learn from my uploaded video?',
      a: 'Not automatically. User uploads are never used for training by default. Only videos that users explicitly approve — and which pass human verification — may be added to a secure training pipeline.',
    },
    {
      q: 'How accurate is the system?',
      a: 'Accuracy depends on video quality and type. The system is designed to prioritize reliability over overconfidence. If the AI is uncertain, it flags the video instead of giving a misleading result.',
    },
    {
      q: 'What happens if the system is unsure?',
      a: 'When confidence is low, the system either runs additional specialist models, or flags the video as "uncertain" instead of forcing a prediction. This reduces false positives and false negatives.',
    },
    {
      q: 'Can this detect all types of deepfakes?',
      a: 'No system can guarantee 100% detection. Our approach is adaptive and extensible, meaning new specialist models can be added as new deepfake techniques emerge.',
    },
    {
      q: 'Does this system analyze audio as well?',
      a: 'Currently, the primary focus is video analysis. Audio-visual consistency checks (such as lip-sync analysis) are part of the planned future extensions.',
    },
    {
      q: 'Can this be used offline?',
      a: 'The current version runs online via a secure server. A future offline mobile and edge-device version is planned for field and low-connectivity environments.',
    },
    {
      q: 'Is this system open-source?',
      a: 'Parts of the system are developed as research components. Full open-source release depends on security and deployment considerations.',
    },
    {
      q: 'Who is this system designed for?',
      a: 'This system is designed for: Law enforcement and investigators, Journalists and fact-checkers, Digital forensics teams, and General users concerned about misinformation.',
    },
    {
      q: 'Can this be integrated into other platforms?',
      a: 'Yes. The system is designed with API-based integration in mind and can be embedded into websites, apps, or internal tools.',
    },
    {
      q: 'Does the system identify people in videos?',
      a: 'No. The system does not perform identity recognition. It only analyzes video authenticity and does not store or infer personal identity information.',
    },
    {
      q: 'Is this a final production system?',
      a: 'This is a research and development prototype created for the E-Raksha Hackathon. Continuous improvements and evaluations are planned.',
    },
    {
      q: 'How is this system kept secure?',
      a: 'Model versions are verified, training data is curated, and feedback is reviewed by humans before being used. This prevents misuse and model poisoning.',
    },
    {
      q: 'Why should I trust the result?',
      a: 'The system provides confidence scores and visual explanations, helping users understand why a video may be flagged rather than relying on blind predictions.',
    },
    {
      q: 'Can I give feedback on a result?',
      a: 'Yes. Users can optionally provide feedback if they know the ground truth. Feedback helps improve future versions after verification.',
    },
    {
      q: 'Who built this system?',
      a: 'This system was developed by a multidisciplinary student team as part of the E-Raksha Hackathon, focusing on AI safety, trust, and real-world deployment.',
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
                className={`overflow-hidden transition-all duration-300 ease-in-out ${
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
      </div>
    </div>
  );
};

export default FAQ;