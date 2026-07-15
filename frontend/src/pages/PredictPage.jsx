import { useState, useEffect } from 'react';
import DrugSearch from '../components/DrugSearch';
import PredictionPanel from '../components/PredictionPanel';
import { api } from '../api/client';

export default function PredictPage() {
  const [results, setResults] = useState(null);
  const [labels, setLabels] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getLabels()
      .then(setLabels)
      .catch(() => {});
  }, []);

  const handlePredict = async (drugA, drugB) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await api.predict(drugA, drugB);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header animate-in">
        <h1 className="page-title">
          🔬 Drug Interaction Prediction
        </h1>
        <p className="page-subtitle">
          Enter two drug SMILES strings to predict their interaction using
          GAT and ChemBERTa models side-by-side.
        </p>
      </div>

      <DrugSearch onPredict={handlePredict} loading={loading} />

      {error && (
        <div className="card animate-in" style={{
          borderColor: 'rgba(239, 68, 68, 0.3)',
          marginBottom: 'var(--space-xl)',
        }}>
          <span style={{ color: 'var(--color-accent-danger)' }}>
            ⚠️ {error}
          </span>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <span>Running predictions across all models...</span>
        </div>
      )}

      {results && <PredictionPanel results={results} labels={labels} />}
    </div>
  );
}
