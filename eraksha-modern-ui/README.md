# E-Raksha Modern UI

A modern React/TypeScript application for deepfake detection with a beautiful, responsive interface.

## Features

- **Modern Tech Stack**: React 18, TypeScript, Vite
- **Styling**: Tailwind CSS with shadcn/ui components
- **Animations**: Framer Motion for smooth interactions
- **Theme Support**: Dark/Light mode toggle
- **Responsive Design**: Works on all devices
- **Real-time Analysis**: Progress tracking and live updates
- **Interactive Upload**: Drag & drop file upload
- **Data Visualization**: Charts and analytics with Recharts

## Quick Start

### Option 1: Use the batch file (Windows)
```bash
# From the root directory
start_modern_website.bat
```

### Option 2: Manual setup
```bash
cd eraksha-modern-ui
npm install
npm run dev
```

The application will be available at `http://localhost:3001`

## Project Structure

```
eraksha-modern-ui/
├── src/
│   ├── components/
│   │   ├── ui/           # shadcn/ui components
│   │   └── Navbar.tsx    # Navigation component
│   ├── pages/
│   │   ├── Home.tsx      # Landing page
│   │   ├── AnalysisWorkbench.tsx  # Video upload & analysis
│   │   ├── Dashboard.tsx # Analysis history
│   │   ├── Analytics.tsx # Statistics & charts
│   │   ├── Contact.tsx   # Contact form
│   │   └── FAQ.tsx       # Frequently asked questions
│   ├── context/
│   │   └── ThemeContext.tsx  # Theme management
│   ├── lib/
│   │   └── utils.ts      # Utility functions
│   ├── App.tsx           # Main app component
│   ├── main.tsx          # Entry point
│   └── index.css         # Global styles
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## API Integration

The Analysis Workbench page is configured to connect to the backend API at `http://localhost:8000/analyze`. Make sure your backend server is running for full functionality.

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Technologies Used

- React 18 with TypeScript
- Vite for fast development
- Tailwind CSS for styling
- shadcn/ui for components
- Framer Motion for animations
- Recharts for data visualization
- React Router for navigation
- Lucide React for icons