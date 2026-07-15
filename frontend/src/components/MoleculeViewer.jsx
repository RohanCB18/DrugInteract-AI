import { renderMoleculeSVG } from '../utils/moleculeRenderer';

export default function MoleculeViewer({ structure, importance, label }) {
  const svgString = renderMoleculeSVG(structure, importance || {}, 380, 220);

  return (
    <div>
      {label && (
        <div style={{
          fontSize: '0.75rem',
          fontWeight: 600,
          color: 'var(--color-text-secondary)',
          marginBottom: 'var(--space-sm)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          {label}
        </div>
      )}
      <div className="molecule-viewer">
        <div dangerouslySetInnerHTML={{ __html: svgString }} />
      </div>
      {importance && Object.keys(importance).length > 0 && (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 'var(--space-lg)',
          marginTop: 'var(--space-sm)',
          fontSize: '0.65rem',
          color: 'var(--color-text-muted)',
        }}>
          <span>🔵 Low importance</span>
          <span>🟣 Medium</span>
          <span>🔴 High importance</span>
        </div>
      )}
    </div>
  );
}
