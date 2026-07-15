import ComparisonDashboard from '../components/ComparisonDashboard';

export default function DashboardPage() {
  return (
    <div>
      <div className="page-header animate-in">
        <h1 className="page-title">📊 Model Comparison Dashboard</h1>
        <p className="page-subtitle">
          Compare accuracy, precision, recall, and F1 scores across all model
          variants — GAT vs. ChemBERTa, baseline vs. pretrained.
        </p>
      </div>
      <ComparisonDashboard />
    </div>
  );
}
