import { NavLink } from 'react-router-dom';

export default function Header() {
  return (
    <nav className="nav">
      <div className="nav-brand">
        <div className="nav-brand-icon">💊</div>
        <span className="nav-brand-text">DrugInteract AI</span>
      </div>
      <ul className="nav-links">
        <li>
          <NavLink
            to="/"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            🔬 Predict
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/dashboard"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            📊 Dashboard
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/history"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            📋 History
          </NavLink>
        </li>
      </ul>
    </nav>
  );
}
