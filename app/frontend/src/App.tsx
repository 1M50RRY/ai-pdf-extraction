import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import {
  Zap,
  History,
  Home,
} from "lucide-react";
import { healthCheck } from "./api";
import { DashboardPage } from "./pages/DashboardPage";
import { HistoryPage } from "./pages/HistoryPage";

function NavigationBar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  // Check backend health on mount
  useEffect(() => {
    healthCheck().then(setIsBackendHealthy);
  }, []);

  const handleHomeClick = () => {
    navigate("/");
  };

  const handleHistoryClick = () => {
    navigate("/history");
  };

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-100">
                PDF Extractor
              </h1>
              <p className="text-xs text-slate-400">MVP Edition</p>
            </div>
          </div>

          {/* Navigation & Backend Status */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleHomeClick}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                location.pathname === "/"
                  ? "text-slate-200 bg-slate-700"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-700"
              }`}
            >
              <Home className="w-4 h-4" />
              <span className="text-sm">Home</span>
            </button>
            <button
              onClick={handleHistoryClick}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                location.pathname === "/history"
                  ? "text-slate-200 bg-slate-700"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-700"
              }`}
            >
              <History className="w-4 h-4" />
              <span className="text-sm">History</span>
            </button>
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isBackendHealthy === true
                    ? "bg-emerald-400"
                    : isBackendHealthy === false
                      ? "bg-red-400"
                      : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-xs text-slate-400">
                {isBackendHealthy === true
                  ? "Backend Connected"
                  : isBackendHealthy === false
                    ? "Backend Offline"
                    : "Checking..."}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function AppContent() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950">
      <NavigationBar />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-8">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-16 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          AI PDF Extraction • MVP Edition • Powered by GPT-4.1
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
