import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SemanticEquivalenceAnalysis } from '../../domain/models/document-equivalence.model';

/**
 * Presentation component visualizing semantic alignment score and divergence points.
 */
@Component({
  selector: 'app-similarity-graph',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="result-card" *ngIf="analysis">
      <div class="header-row">
        <div>
          <h4>Resultado da Auditoria Semântica</h4>
          <span class="subtext">ID da Análise: <code>{{ analysis.analysisId }}</code></span>
        </div>
        <div class="badge" [ngClass]="analysis.status.toLowerCase()">
          {{ analysis.status }}
        </div>
      </div>

      <div class="score-container">
        <div class="score-circle">
          <span class="score-value">{{ (analysis.overallScore * 100) | number:'1.0-0' }}%</span>
          <span class="score-label">Similaridade</span>
        </div>
        <div class="summary-box">
          <p class="user-msg">{{ analysis.userMessage }}</p>
          <p class="explanation">{{ analysis.explanation }}</p>
        </div>
      </div>

      <div class="divergence-section" *ngIf="analysis.divergencePoints.length > 0">
        <h5>Pontos de Divergência Identificados:</h5>
        <ul>
          <li *ngFor="let point of analysis.divergencePoints">
            <span class="bullet">⚠️</span> {{ point }}
          </li>
        </ul>
      </div>
    </div>
  `,
  styles: [`
    .result-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px;
      margin-top: 24px;
    }
    .header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .subtext { font-size: 0.8rem; color: var(--text-muted); }
    .badge {
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .badge.equivalent { background: rgba(16, 185, 129, 0.2); color: var(--status-equivalent); border: 1px solid var(--status-equivalent); }
    .badge.partially_equivalent { background: rgba(245, 158, 11, 0.2); color: var(--status-partial); border: 1px solid var(--status-partial); }
    .badge.divergent { background: rgba(239, 68, 68, 0.2); color: var(--status-divergent); border: 1px solid var(--status-divergent); }
    .score-container {
      display: flex;
      gap: 24px;
      align-items: center;
      background: var(--bg-card);
      padding: 20px;
      border-radius: 8px;
    }
    .score-circle {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 100px;
      height: 100px;
      border-radius: 50%;
      border: 4px solid var(--accent-cyan);
    }
    .score-value { font-size: 1.5rem; font-weight: 700; color: var(--accent-cyan); }
    .score-label { font-size: 0.7rem; color: var(--text-muted); }
    .user-msg { font-weight: 600; color: var(--text-main); margin-bottom: 6px; }
    .explanation { font-size: 0.9rem; color: var(--text-muted); }
    .divergence-section { margin-top: 20px; }
    .divergence-section h5 { font-size: 0.95rem; margin-bottom: 8px; color: var(--status-partial); }
    ul { list-style: none; }
    li { font-size: 0.85rem; margin-bottom: 6px; color: var(--text-main); }
  `]
})
export class SimilarityGraphComponent {
  @Input() analysis: SemanticEquivalenceAnalysis | null = null;
}
