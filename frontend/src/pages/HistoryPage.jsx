import RecentPredictions from '../components/RecentPredictions';

export default function HistoryPage() {
  return (
    <div>
      <div className="page-header animate-in">
        <h1 className="page-title">📋 Prediction History</h1>
        <p className="page-subtitle">
          Browse recent drug interaction predictions logged by the system.
        </p>
      </div>
      <RecentPredictions />
    </div>
  );
}
