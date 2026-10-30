import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { UploadPanelComponent } from '../../components/upload-panel/upload-panel.component';
import { SimilarityGraphComponent } from '../../components/similarity-graph/similarity-graph.component';
import { EquivalenceApiService } from '../../../infrastructure/api/equivalence-api.service';
import { SemanticEquivalenceAnalysis } from '../../../domain/models/document-equivalence.model';

/**
 * Main dashboard container orchestrating uploads and displaying equivalence insights.
 */
@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, UploadPanelComponent, SimilarityGraphComponent],
  template: `
    <div class="dashboard-shell">
      <header class="navbar">
        <div class="brand">
          <span class="logo">⚡</span>
          <h2>SemanticDoc Equivalence Engine</h2>
        </div>
        <div class="status-indicator">
          <span class="dot"></span>
          <span>vLLM + Qdrant Ativos</span>
        </div>
      </header>

      <main class="content-body">
        <app-upload-panel (fileSubmitted)="handleFileUpload($event)"></app-upload-panel>

        <div *ngIf="isLoading" class="loading-state">
          <div class="spinner"></div>
          <p>Executando OCR via PaddleOCR, geração de embeddings e síntese LangGraph...</p>
        </div>

        <app-similarity-graph [analysis]="currentAnalysis"></app-similarity-graph>
      </main>
    </div>
  `,
  styles: [`
    .dashboard-shell { max-width: 1000px; margin: 0 auto; padding: 32px 20px; }
    .navbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 32px;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 20px;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo { font-size: 1.8rem; }
    h2 { font-size: 1.4rem; font-weight: 700; }
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--status-equivalent);
      background: rgba(16, 185, 129, 0.1);
      padding: 6px 12px;
      border-radius: 20px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--status-equivalent); }
    .loading-state {
      text-align: center;
      padding: 40px;
      background: var(--bg-surface);
      border-radius: 12px;
      border: 1px solid var(--border-color);
    }
    .spinner {
      width: 36px;
      height: 36px;
      border: 3px solid var(--border-color);
      border-top-color: var(--accent-cyan);
      border-radius: 50%;
      margin: 0 auto 16px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  `]
})
export class DashboardComponent {
  currentAnalysis: SemanticEquivalenceAnalysis | null = null;
  isLoading = false;

  constructor(private equivalenceApi: EquivalenceApiService) {}

  handleFileUpload(file: File) {
    this.isLoading = true;
    this.currentAnalysis = null;

    this.equivalenceApi.analyzeDocument(file).subscribe({
      next: (analysis) => {
        this.currentAnalysis = analysis;
        this.isLoading = false;
      },
      error: (err) => {
        console.error('Erro durante processamento da análise', err);
        this.isLoading = false;
      }
    });
  }
}
