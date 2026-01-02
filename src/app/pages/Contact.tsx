import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, MessageSquare, Send, FileText, Github } from 'lucide-react';

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Create email template
    const emailSubject = encodeURIComponent(formData.subject || 'Contact Form Submission');
    const emailBody = encodeURIComponent(
      `Dear Interceptor Team,

I hope this message finds you well. I am reaching out regarding ${formData.subject.toLowerCase()}.

${formData.message}

I would appreciate your assistance with this matter. Please feel free to contact me at your earliest convenience.

Best regards,
${formData.name}

---
Contact Information:
Name: ${formData.name}
Email: ${formData.email}
Subject: ${formData.subject}

This message was sent via the Interceptor contact form.`
    );
    
    // Open email client with pre-filled template
    window.location.href = `mailto:help.interceptor@gmail.com?subject=${emailSubject}&body=${emailBody}`;
  };

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl md:text-4xl mb-3 font-bold text-gray-900 dark:text-white">
            Contact Us
          </h1>
          <p className="text-base text-gray-600 dark:text-gray-400">
            Have questions or feedback? We'd love to hear from you.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Contact Form */}
          <div className="rounded-xl p-8 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800">
            <h2 className="text-xl mb-6 font-bold text-gray-900 dark:text-white">
              Send us a message
            </h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm mb-2 text-gray-700 dark:text-gray-300">
                  Name
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  required
                  className="w-full px-4 py-3 rounded-lg border transition-all bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-blue-400/20"
                  placeholder="Your name"
                />
              </div>
              <div>
                <label className="block text-sm mb-2 text-gray-700 dark:text-gray-300">
                  Email
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  required
                  className="w-full px-4 py-3 rounded-lg border transition-all bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-blue-400/20"
                  placeholder="your.email@example.com"
                />
              </div>
              <div>
                <label className="block text-sm mb-2 text-gray-700 dark:text-gray-300">
                  Subject
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleInputChange}
                  required
                  className="w-full px-4 py-3 rounded-lg border transition-all bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-blue-400/20"
                  placeholder="What's this about?"
                />
              </div>
              <div>
                <label className="block text-sm mb-2 text-gray-700 dark:text-gray-300">
                  Message
                </label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleInputChange}
                  required
                  rows={6}
                  className="w-full px-4 py-3 rounded-lg border transition-all resize-none bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white focus:border-blue-500 dark:focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-blue-400/20"
                  placeholder="Tell us more..."
                />
              </div>
              <button
                type="submit"
                className="w-full px-6 py-4 rounded-lg transition-all flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white"
              >
                <Send className="w-5 h-5" />
                Send Message
              </button>
            </form>
          </div>

          {/* Contact Info */}
          <div className="space-y-6">
            {/* Email */}
            <div className="rounded-xl p-6 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-blue-500/20">
                  <Mail className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg mb-1 font-bold text-gray-900 dark:text-white">
                    Email
                  </h3>
                  <p className="text-sm mb-2 text-gray-600 dark:text-gray-400">
                    Send us an email anytime
                  </p>
                  <a
                    href="mailto:help.interceptor@gmail.com"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
                  >
                    help.interceptor@gmail.com
                  </a>
                </div>
              </div>
            </div>

            {/* Documentation */}
            <div className="rounded-xl p-6 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-blue-500/20">
                  <FileText className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg mb-1 font-bold text-gray-900 dark:text-white">
                    Documentation
                  </h3>
                  <p className="text-sm mb-2 text-gray-600 dark:text-gray-400">
                    Read our guides and API docs
                  </p>
                  <a
                    href="https://docs.google.com/document/d/1hYk1rtjOidR3H3cWOa9RS3BBEQ7n1wQGxVMzI3QYfwM/edit?usp=sharing"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
                  >
                    Google Docs
                  </a>
                </div>
              </div>
            </div>

            {/* GitHub */}
            <div className="rounded-xl p-6 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-blue-500/20">
                  <Github className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg mb-1 font-bold text-gray-900 dark:text-white">
                    GitHub
                  </h3>
                  <p className="text-sm mb-2 text-gray-600 dark:text-gray-400">
                    Check out our code and contribute
                  </p>
                  <a
                    href="https://github.com/Pranay22077/deepfake-agentic"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
                  >
                    View Repository
                  </a>
                </div>
              </div>
            </div>

            {/* FAQ */}
            <div className="rounded-xl p-6 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-blue-500/20">
                  <MessageSquare className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg mb-1 font-bold text-gray-900 dark:text-white">
                    FAQ
                  </h3>
                  <p className="text-sm mb-2 text-gray-600 dark:text-gray-400">
                    Find answers to common questions in our FAQ section
                  </p>
                  <Link
                    to="/faq"
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
                  >
                    View All FAQs →
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;
