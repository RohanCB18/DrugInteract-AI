import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import PredictPage from './pages/PredictPage';
import DashboardPage from './pages/DashboardPage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<PredictPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
