/**
 * moleculeRenderer.js — Parse backend molecule data and render as SVG.
 *
 * Renders atoms as colored circles with element labels, bonds as lines,
 * and overlays attention-based importance as a blue→red heatmap.
 */

const ELEMENT_COLORS = {
  C: '#a0a0a0',
  N: '#3b82f6',
  O: '#ef4444',
  S: '#eab308',
  F: '#22c55e',
  Cl: '#22c55e',
  Br: '#a855f7',
  P: '#f97316',
  I: '#7c3aed',
  H: '#d1d5db',
};

const BOND_WIDTHS = {
  SINGLE: 2,
  DOUBLE: 2,
  TRIPLE: 2,
  AROMATIC: 2,
};

/**
 * Convert importance score (0-1) to a color on a blue → red heatmap.
 */
function importanceToColor(value) {
  const r = Math.round(255 * value);
  const b = Math.round(255 * (1 - value));
  const g = Math.round(80 * (1 - Math.abs(value - 0.5) * 2));
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Render molecule structure as SVG string.
 *
 * @param {Object} structure - { atoms: [...], bonds: [...] }
 * @param {Object} importance - { atom_idx: score } (0-1 range)
 * @param {number} width
 * @param {number} height
 * @returns {string} SVG markup
 */
export function renderMoleculeSVG(structure, importance = {}, width = 400, height = 250) {
  if (!structure || !structure.atoms || structure.atoms.length === 0) {
    return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <text x="50%" y="50%" text-anchor="middle" fill="#64748b" font-size="14">
        No structure data
      </text>
    </svg>`;
  }

  const { atoms, bonds } = structure;
  const padding = 40;

  // Calculate bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  atoms.forEach(a => {
    minX = Math.min(minX, a.x);
    maxX = Math.max(maxX, a.x);
    minY = Math.min(minY, a.y);
    maxY = Math.max(maxY, a.y);
  });

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const scale = Math.min(
    (width - 2 * padding) / rangeX,
    (height - 2 * padding) / rangeY
  );

  const transform = (x, y) => ({
    x: padding + (x - minX) * scale,
    y: padding + (y - minY) * scale,
  });

  let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background: transparent;">`;

  // Draw bonds
  bonds.forEach(bond => {
    const a1 = atoms.find(a => a.idx === bond.begin);
    const a2 = atoms.find(a => a.idx === bond.end);
    if (!a1 || !a2) return;

    const p1 = transform(a1.x, a1.y);
    const p2 = transform(a2.x, a2.y);
    const bw = BOND_WIDTHS[bond.type] || 2;

    if (bond.type === 'DOUBLE') {
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const ox = (-dy / len) * 2.5;
      const oy = (dx / len) * 2.5;

      svg += `<line x1="${p1.x + ox}" y1="${p1.y + oy}" x2="${p2.x + ox}" y2="${p2.y + oy}" stroke="#4b5563" stroke-width="${bw}" />`;
      svg += `<line x1="${p1.x - ox}" y1="${p1.y - oy}" x2="${p2.x - ox}" y2="${p2.y - oy}" stroke="#4b5563" stroke-width="${bw}" />`;
    } else if (bond.type === 'TRIPLE') {
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const ox = (-dy / len) * 3;
      const oy = (dx / len) * 3;

      svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="#4b5563" stroke-width="${bw}" />`;
      svg += `<line x1="${p1.x + ox}" y1="${p1.y + oy}" x2="${p2.x + ox}" y2="${p2.y + oy}" stroke="#4b5563" stroke-width="${bw}" />`;
      svg += `<line x1="${p1.x - ox}" y1="${p1.y - oy}" x2="${p2.x - ox}" y2="${p2.y - oy}" stroke="#4b5563" stroke-width="${bw}" />`;
    } else if (bond.type === 'AROMATIC') {
      svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="#6366f1" stroke-width="${bw}" stroke-dasharray="4 2" />`;
    } else {
      svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="#4b5563" stroke-width="${bw}" />`;
    }
  });

  // Draw atoms
  atoms.forEach(atom => {
    const p = transform(atom.x, atom.y);
    const imp = importance[atom.idx] ?? 0;
    const radius = 14 + imp * 6;
    const fill = imp > 0 ? importanceToColor(imp) : (ELEMENT_COLORS[atom.symbol] || '#9ca3af');
    const opacity = imp > 0 ? 0.3 + imp * 0.5 : 0.15;

    // Glow ring for important atoms
    if (imp > 0.3) {
      svg += `<circle cx="${p.x}" cy="${p.y}" r="${radius + 4}" fill="none" stroke="${importanceToColor(imp)}" stroke-width="1.5" opacity="${imp * 0.5}" />`;
    }

    // Atom circle
    svg += `<circle cx="${p.x}" cy="${p.y}" r="${radius}" fill="${fill}" opacity="${opacity}" />`;

    // Element label
    svg += `<text x="${p.x}" y="${p.y + 4}" text-anchor="middle" fill="#f1f5f9" font-size="10" font-weight="600" font-family="Inter, sans-serif">${atom.symbol}</text>`;
  });

  svg += '</svg>';
  return svg;
}
