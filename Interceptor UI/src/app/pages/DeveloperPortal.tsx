import React from 'react';
import { Code, Book, Zap, Copy, ExternalLink } from 'lucide-react';

const DeveloperPortal = () => {
  const apiEndpoints = [
    {
      method: 'POST',
      path: '/predict',
      description: 'Main video analysis endpoint',
    },
    {
      method: 'GET',
      path: '/health',
      description: 'System health check',
    },
    {
      method: 'GET',
      path: '/stats',
      description: 'System statistics',
    },
    {
      method: 'POST',
      path: '/feedback',
      description: 'Submit user corrections',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-purple-950/20 to-gray-950 pt-24 pb-20">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Integrate Deepfake Detection Into Your Workflow
          </h1>
          <p className="text-gray-400 max-w-3xl">
            Add powerful AI-driven deepfake detection to your applications with our REST API. Comprehensive documentation, SDKs, and live testing tools.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="w-12 h-12 bg-blue-600/20 rounded-lg flex items-center justify-center text-blue-400 mb-4">
              <Code className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">REST API Access</h3>
            <p className="text-sm text-gray-400">
              Simple HTTP endpoints for video analysis, stats, and feedback
            </p>
          </div>

          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center text-purple-400 mb-4">
              <Book className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">OpenAPI/Swagger Docs</h3>
            <p className="text-sm text-gray-400">
              Complete API documentation with request/response examples
            </p>
          </div>

          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <div className="w-12 h-12 bg-green-600/20 rounded-lg flex items-center justify-center text-green-400 mb-4">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Python & JavaScript SDKs</h3>
            <p className="text-sm text-gray-400">
              Official SDKs with full type support and examples
            </p>
          </div>
        </div>

        {/* API Playground */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8 mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">
            Interactive API Playground
          </h2>
          <p className="text-gray-400 mb-6">
            Test endpoints with Swagger UI for live testing and experimentation
          </p>

          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">API Endpoints</h3>
              <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors text-sm">
                <ExternalLink className="w-4 h-4" />
                Open Swagger UI
              </button>
            </div>

            <div className="space-y-3">
              {apiEndpoints.map((endpoint, index) => (
                <div
                  key={index}
                  className="bg-gray-900/50 border border-gray-700 rounded-lg p-4 hover:border-purple-500 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <span
                      className={`px-3 py-1 rounded text-xs font-semibold ${
                        endpoint.method === 'POST'
                          ? 'bg-green-600/20 text-green-400'
                          : 'bg-blue-600/20 text-blue-400'
                      }`}
                    >
                      {endpoint.method}
                    </span>
                    <code className="text-sm text-purple-400">{endpoint.path}</code>
                    <span className="text-sm text-gray-400 flex-1">{endpoint.description}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Documentation */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8 mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">
            Comprehensive API Documentation
          </h2>
          
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Quick Start Example</h3>
            
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-400 mb-2">Python</p>
                <div className="bg-gray-950 border border-gray-700 rounded-lg p-4 relative">
                  <button className="absolute top-2 right-2 p-2 hover:bg-gray-800 rounded transition-colors">
                    <Copy className="w-4 h-4 text-gray-400" />
                  </button>
                  <pre className="text-sm text-gray-300 overflow-x-auto">
                    <code>{`import requests

url = "http://localhost:8000/predict"
files = {"file": open("video.mp4", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")`}</code>
                  </pre>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-400 mb-2">JavaScript</p>
                <div className="bg-gray-950 border border-gray-700 rounded-lg p-4 relative">
                  <button className="absolute top-2 right-2 p-2 hover:bg-gray-800 rounded transition-colors">
                    <Copy className="w-4 h-4 text-gray-400" />
                  </button>
                  <pre className="text-sm text-gray-300 overflow-x-auto">
                    <code>{`const formData = new FormData();
formData.append('file', videoFile);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Prediction:', result.prediction);
console.log('Confidence:', result.confidence);`}</code>
                  </pre>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-gray-700">
              <h4 className="text-sm font-semibold text-white mb-3">Response Format</h4>
              <div className="bg-gray-950 border border-gray-700 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
                  <code>{`{
  "prediction": "fake",
  "confidence": 0.73,
  "faces_analyzed": 5,
  "analysis": {
    "confidence_breakdown": {
      "raw_confidence": 0.73,
      "quality_adjusted": 0.68
    }
  }
}`}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>

        {/* Pricing */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8 mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">
            Clear developer API usage pricing tiers
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-2">Free Tier</h3>
              <p className="text-3xl font-bold text-white mb-4">$0</p>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>• 100 API calls/month</li>
                <li>• Basic support</li>
                <li>• Community access</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-2">Pro</h3>
              <p className="text-3xl font-bold text-white mb-4">Contact Sales</p>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Unlimited API calls</li>
                <li>• Priority support</li>
                <li>• Custom integration</li>
                <li>• SLA guarantees</li>
              </ul>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8 mb-12">
          <h2 className="text-2xl font-bold text-white mb-6">API Integration FAQs</h2>
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">What are the rate limits?</h3>
              <p className="text-gray-400">
                Free tier: 100 calls/month. Pro tier has no rate limits with SLA guarantees.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">How is data handled?</h3>
              <p className="text-gray-400">
                All uploaded videos are processed in-memory and deleted immediately after analysis. We don't store your data.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">What support channels are available?</h3>
              <p className="text-gray-400">
                Free tier: Community forum. Pro tier: Email support, Slack integration, and dedicated account manager.
              </p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-800/50 rounded-2xl p-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">
            Join our growing API developer community
          </h2>
          <p className="text-gray-300 mb-6">
            Get started with our API or join our developer Slack for support and collaboration
          </p>
          <div className="flex items-center justify-center gap-4">
            <button className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
              Get Started with API
            </button>
            <button className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors">
              Join Dev Slack
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeveloperPortal;
