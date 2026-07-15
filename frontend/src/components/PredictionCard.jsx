export default function PredictionCard({ prediction, labels, groundTruth }) {
  if (!prediction) return null;

  const { model_name, prediction: predClass, predicted_label, confidence, probabilities, error } = prediction;

  const isGAT = model_name?.startsWith('GAT');
  const version = model_name?.includes('-A') ? 'A' : 'B';

  if (error) {
    return (
      <div className={`prediction-card ${isGAT ? 'gat' : 'chemberta'} animate-in`}>
        <div className="card-header">
          <div>
            <span className={`badge ${isGAT ? 'badge-gat' : 'badge-chemberta'}`}>
              {isGAT ? 'GAT' : 'ChemBERTa'}
            </span>
            <span className={`badge ${version === 'A' ? 'badge-version-a' : 'badge-version-b'}`} style={{ marginLeft: 6 }}>
              V{version}
            </span>
          </div>
        </div>
        <div style={{ color: 'var(--color-accent-danger)', fontSize: '0.85rem' }}>
          ⚠️ {error}
        </div>
      </div>
    );
  }

  const labelNames = labels || {};
  const allLabels = Object.values(labelNames);

  return (
    <div className={`prediction-card ${isGAT ? 'gat' : 'chemberta'} animate-in`}>
      {/* Header */}
      <div className="card-header">
        <div style={{ display: 'flex', gap: 6 }}>
          <span className={`badge ${isGAT ? 'badge-gat' : 'badge-chemberta'}`}>
            {isGAT ? '🔷 GAT' : '🔶 ChemBERTa'}
          </span>
          <span className={`badge ${version === 'A' ? 'badge-version-a' : 'badge-version-b'}`}>
            {version === 'A' ? 'Baseline' : 'Pretrained'}
          </span>
        </div>
      </div>

      {/* Predicted class */}
      <div className="prediction-class">
        {predicted_label || `Class ${predClass}`}
      </div>

      {/* Confidence */}
      <div className="prediction-confidence">
        Confidence: {(confidence * 100).toFixed(1)}%
      </div>
      <div className="confidence-bar-container" style={{ marginBottom: groundTruth !== undefined && groundTruth !== null ? '0.5rem' : '0' }}>
        <div
          className="confidence-bar"
          style={{ width: `${confidence * 100}%` }}
        />
      </div>

      {/* Accuracy Status Indicator */}
      {groundTruth !== undefined && groundTruth !== null && (
        <div style={{
          marginBottom: 'var(--space-md)',
          fontSize: '0.75rem',
          fontWeight: 700,
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          color: groundTruth === predClass ? '#10b981' : '#ef4444',
          textTransform: 'uppercase',
          letterSpacing: '0.05em'
        }}>
          {groundTruth === predClass ? (
            <><span>✅</span> Correct prediction</>
          ) : (
            <><span>❌</span> Incorrect (Truth: {labelNames[groundTruth] || `Class ${groundTruth}`})</>
          )}
        </div>
      )}

      {/* Probability breakdown */}
      {probabilities && probabilities.length > 0 && (
        <div className="prediction-probs">
          {probabilities.map((prob, i) => {
            const label = allLabels[i] || `Class ${i}`;
            const barColor = i === predClass
              ? (isGAT ? 'var(--color-accent-1)' : 'var(--color-accent-3)')
              : 'var(--color-text-muted)';
            return (
              <div className="prob-row" key={i}>
                <span className="prob-label" title={label}>{label}</span>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar"
                    style={{
                      width: `${prob * 100}%`,
                      background: barColor,
                      opacity: i === predClass ? 1 : 0.4,
                    }}
                  />
                </div>
                <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
