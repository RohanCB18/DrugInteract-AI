import { useState, useEffect } from 'react';
import { api } from '../api/client';

export default function RecentPredictions() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getHistory(100)
      .then(data => setPredictions(data || []))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <span>Loading prediction history...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">⚠️</div>
        <div className="empty-state-title">Could not load history</div>
        <p style={{ color: 'var(--color-text-muted)' }}>{error}</p>
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📋</div>
        <div className="empty-state-title">No predictions yet</div>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Make a prediction to see it logged here.
        </p>
      </div>
    );
  }

  const formatTime = (iso) => {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  return (
    <div className="card animate-in">
      <div className="card-header">
        <h3 className="card-title">📋 Recent Predictions</h3>
        <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
          {predictions.length} entries
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Drug A</th>
              <th>Drug B</th>
              <th>Model</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => (
              <tr key={p.id}>
                <td style={{ color: 'var(--color-text-muted)' }}>{p.id}</td>
                <td title={p.smiles_a}>{p.drug_a}</td>
                <td title={p.smiles_b}>{p.drug_b}</td>
                <td>
                  <span className={`badge ${p.model_name?.startsWith('GAT') ? 'badge-gat' : 'badge-chemberta'}`}>
                    {p.model_name}
                  </span>
                </td>
                <td>{p.predicted_label || `Class ${p.predicted_class}`}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>
                  {(p.confidence * 100).toFixed(1)}%
                </td>
                <td style={{ color: 'var(--color-text-muted)' }}>
                  {formatTime(p.timestamp)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
