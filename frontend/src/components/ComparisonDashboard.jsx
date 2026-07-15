import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import { api } from '../api/client';

const MODEL_COLORS = {
  'GAT': '#6366f1',
  'ChemBERTa': '#10b981',
};

export default function ComparisonDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getComparison()
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <span>Loading evaluation metrics...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">⚠️</div>
        <div className="empty-state-title">Could not load metrics</div>
        <p style={{ color: 'var(--color-text-muted)' }}>{error}</p>
      </div>
    );
  }

  if (!data || !data.models || data.models.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <div className="empty-state-title">No evaluation results yet</div>
        <p style={{ color: 'var(--color-text-muted)' }}>
          Train models and run the evaluation script to see comparisons.
        </p>
      </div>
    );
  }

  const { models } = data;

  // Prepare bar chart data
  const metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'];
  const metricLabels = {
    accuracy: 'Accuracy',
    precision_macro: 'Precision',
    recall_macro: 'Recall',
    f1_macro: 'F1 Score',
  };

  const barData = metrics.map(metric => {
    const row = { metric: metricLabels[metric] };
    models.forEach(m => {
      row[m.model_name] = m[metric];
    });
    return row;
  });

  // Prepare radar chart data
  const radarData = metrics.map(metric => {
    const row = { metric: metricLabels[metric] };
    models.forEach(m => {
      row[m.model_name] = m[metric] * 100;
    });
    return row;
  });

  // Find best model for each metric
  const bestByMetric = {};
  metrics.forEach(metric => {
    let best = null;
    let bestVal = -1;
    models.forEach(m => {
      if (m[metric] > bestVal) {
        bestVal = m[metric];
        best = m.model_name;
      }
    });
    bestByMetric[metric] = best;
  });

  return (
    <div>
      {/* Comparison table */}
      <div className="card animate-in" style={{ marginBottom: 'var(--space-2xl)' }}>
        <div className="card-header">
          <h3 className="card-title">📋 Performance Comparison</h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
            Test set: {data.test_size?.toLocaleString()} samples
          </span>
        </div>
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 (Macro)</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m.model_name}>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge ${m.model_name.startsWith('GAT') ? 'badge-gat' : 'badge-chemberta'}`}>
                      {m.model_name}
                    </span>
                  </div>
                </td>
                {metrics.map(metric => (
                  <td key={metric}>
                    <span className={`metric-value ${bestByMetric[metric] === m.model_name ? 'metric-best' : ''}`}>
                      {(m[metric] * 100).toFixed(1)}%
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts */}
      <div className="grid-2">
        {/* Bar chart */}
        <div className="chart-container animate-in">
          <h3 className="card-title" style={{ marginBottom: 'var(--space-md)' }}>
            📊 Metric Comparison
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} barGap={2}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9',
                }}
                formatter={v => `${(v * 100).toFixed(1)}%`}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              {models.map(m => (
                <Bar
                  key={m.model_name}
                  dataKey={m.model_name}
                  fill={MODEL_COLORS[m.model_name] || '#6366f1'}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar chart */}
        <div className="chart-container animate-in">
          <h3 className="card-title" style={{ marginBottom: 'var(--space-md)' }}>
            🎯 Model Strengths
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(148,163,184,0.15)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <PolarRadiusAxis
                domain={[0, 100]}
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickFormatter={v => `${v}%`}
              />
              {models.map(m => (
                <Radar
                  key={m.model_name}
                  name={m.model_name}
                  dataKey={m.model_name}
                  stroke={MODEL_COLORS[m.model_name] || '#6366f1'}
                  fill={MODEL_COLORS[m.model_name] || '#6366f1'}
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
              ))}
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9',
                }}
                formatter={v => `${v.toFixed(1)}%`}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
