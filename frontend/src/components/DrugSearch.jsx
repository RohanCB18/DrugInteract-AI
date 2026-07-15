import { useState, useEffect } from 'react';
import { TEST_SET_PAIRS } from '../utils/test_pairs';

export default function DrugSearch({ onPredict, loading }) {
  const [customMode, setCustomMode] = useState(false);
  const [selectedPairIndex, setSelectedPairIndex] = useState('');
  
  // Custom mode SMILES states
  const [customA, setCustomA] = useState('');
  const [customB, setCustomB] = useState('');

  // Selected test set pair states
  const [smilesA, setSmilesA] = useState('');
  const [smilesB, setSmilesB] = useState('');
  const [nameA, setNameA] = useState('');
  const [nameB, setNameB] = useState('');

  // Update selected pair details when index changes
  useEffect(() => {
    if (selectedPairIndex !== '') {
      const pair = TEST_SET_PAIRS[parseInt(selectedPairIndex, 10)];
      if (pair) {
        setSmilesA(pair.smilesA);
        setSmilesB(pair.smilesB);
        setNameA(pair.a);
        setNameB(pair.b);
      }
    } else {
      setSmilesA('');
      setSmilesB('');
      setNameA('');
      setNameB('');
    }
  }, [selectedPairIndex]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const finalA = customMode ? customA : smilesA;
    const finalB = customMode ? customB : smilesB;
    if (finalA.trim() && finalB.trim()) {
      onPredict(finalA.trim(), finalB.trim());
    }
  };

  return (
    <div className="drug-search-section card animate-in" style={{ padding: 'var(--space-xl)', background: 'var(--color-bg-secondary)' }}>
      <form onSubmit={handleSubmit}>
        {!customMode ? (
          /* ========================================== */
          /* STANDARD MODE: 25 TEST SET PAIRS SELECTOR */
          /* ========================================== */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            <div className="input-group">
              <label className="input-label" htmlFor="test-pair-select" style={{ fontSize: '0.85rem', color: 'var(--color-accent-1)' }}>
                🎯 Select a Verified Test Set Pair (25 Options)
              </label>
              <select
                id="test-pair-select"
                className="input"
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.95rem',
                  padding: '0.85rem 1.25rem',
                  cursor: 'pointer',
                  background: 'rgba(99, 102, 241, 0.04)',
                  borderColor: 'rgba(99, 102, 241, 0.25)',
                  borderRadius: 'var(--radius-md)'
                }}
                value={selectedPairIndex}
                onChange={(e) => setSelectedPairIndex(e.target.value)}
                disabled={loading}
              >
                <option value="" disabled>-- Click here to select a pair --</option>
                {TEST_SET_PAIRS.map((pair, idx) => (
                  <option key={idx} value={idx}>
                    {idx + 1}. {pair.a} × {pair.b} ({pair.label.split(' (')[0]})
                  </option>
                ))}
              </select>
            </div>

            {selectedPairIndex !== '' && (
              <div className="animate-in" style={{
                padding: 'var(--space-md) var(--space-lg)',
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-sm)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: 'var(--space-xs)' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', fontWeight: 600 }}>DRUG A</span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', fontWeight: 600 }}>DRUG B</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 'var(--space-lg)' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 700, color: 'var(--color-text-primary)', fontSize: '1.05rem' }}>{nameA}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--color-text-muted)', wordBreak: 'break-all', marginTop: 4 }}>
                      {smilesA}
                    </div>
                  </div>
                  <div style={{ fontSize: '1.25rem', color: 'var(--color-text-muted)' }}>×</div>
                  <div style={{ flex: 1, textAlign: 'right' }}>
                    <div style={{ fontWeight: 700, color: 'var(--color-text-primary)', fontSize: '1.05rem' }}>{nameB}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--color-text-muted)', wordBreak: 'break-all', marginTop: 4 }}>
                      {smilesB}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* ========================================== */
          /* CUSTOM MODE: ENTER SMILES MANUALLY       */
          /* ========================================== */
          <div className="drug-inputs animate-in">
            <div className="input-group">
              <label className="input-label" htmlFor="custom-a-input">Drug A — SMILES</label>
              <input
                id="custom-a-input"
                className="input"
                type="text"
                placeholder="Paste structure (e.g. CC(=O)OC...)"
                value={customA}
                onChange={(e) => setCustomA(e.target.value)}
                disabled={loading}
              />
            </div>
            
            <div className="drug-separator">×</div>

            <div className="input-group">
              <label className="input-label" htmlFor="custom-b-input">Drug B — SMILES</label>
              <input
                id="custom-b-input"
                className="input"
                type="text"
                placeholder="Paste structure (e.g. CC(=O)NC...)"
                value={customB}
                onChange={(e) => setCustomB(e.target.value)}
                disabled={loading}
              />
            </div>
          </div>
        )}

        {/* Form Actions */}
        <div style={{ marginTop: 'var(--space-lg)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-md)' }}>
          <button
            type="submit"
            className="btn btn-primary btn-lg"
            style={{ width: '100%', maxWidth: '320px' }}
            disabled={
              loading ||
              (customMode ? (!customA.trim() || !customB.trim()) : !smilesA.trim())
            }
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }}></span>
                Running model inference...
              </>
            ) : (
              <>🧬 Predict Interaction</>
            )}
          </button>

          <button
            type="button"
            className="custom-smiles-toggle"
            style={{ fontSize: '0.8rem', borderBottom: '1px dashed' }}
            onClick={() => {
              setCustomMode(!customMode);
              setSelectedPairIndex('');
              setCustomA('');
              setCustomB('');
            }}
          >
            {customMode ? "Switch to Test Set Dropdown Selector" : "🧪 Or enter custom SMILES strings manually"}
          </button>
        </div>
      </form>
    </div>
  );
}
