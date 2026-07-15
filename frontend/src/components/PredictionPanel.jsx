import PredictionCard from './PredictionCard';
import MoleculeViewer from './MoleculeViewer';

export default function PredictionPanel({ results, labels }) {
  if (!results) return null;

  const { drug_a, drug_b, predictions } = results;

  // Pick attention data from first GAT and first ChemBERTa model
  const gatPred = predictions.find(p => p.model_name?.startsWith('GAT') && !p.error);
  const chembertaPred = predictions.find(p => p.model_name?.startsWith('ChemBERTa') && !p.error);

  // Use GAT attention for molecule highlighting (atom-level is more meaningful)
  const attentionA = gatPred?.attention_a || {};
  const attentionB = gatPred?.attention_b || {};

  return (
    <div className="animate-in">
      {/* Molecule structures with attention highlights */}
      <div className="section-title">
        <div className="section-title-icon">🧬</div>
        Molecule Structures
      </div>
      <div className="grid-2" style={{ marginBottom: 'var(--space-2xl)' }}>
        <MoleculeViewer
          structure={drug_a?.structure}
          importance={attentionA}
          label="Drug A"
        />
        <MoleculeViewer
          structure={drug_b?.structure}
          importance={attentionB}
          label="Drug B"
        />
      </div>

      {/* Ground Truth Banner */}
      {results.ground_truth_label && (
        <div className="card animate-in" style={{
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%)',
          borderColor: 'rgba(16, 185, 129, 0.3)',
          marginBottom: 'var(--space-xl)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
            <span style={{ fontSize: '1.5rem' }}>🎯</span>
            <div>
              <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--color-accent-4)', fontWeight: 700, letterSpacing: '0.05em' }}>
                Clinical Ground Truth (from dataset split)
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--color-text-primary)', marginTop: '2px' }}>
                {results.ground_truth_label}
              </div>
            </div>
          </div>
          <span className="badge badge-version-b" style={{ fontSize: '0.75rem', padding: '0.3rem 0.8rem', background: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}>
            Verified Sample
          </span>
        </div>
      )}

      {/* Prediction cards */}
      <div className="section-title">
        <div className="section-title-icon">🤖</div>
        Model Predictions
      </div>
      <div className="grid-2x2">
        {predictions.map((pred, i) => (
          <PredictionCard
            key={pred.model_name || i}
            prediction={pred}
            labels={labels}
            groundTruth={results.ground_truth}
          />
        ))}
      </div>

      {/* No predictions available */}
      {predictions.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">🤷</div>
          <div className="empty-state-title">No models loaded</div>
          <p>Train at least one model to see predictions.</p>
        </div>
      )}
    </div>
  );
}
