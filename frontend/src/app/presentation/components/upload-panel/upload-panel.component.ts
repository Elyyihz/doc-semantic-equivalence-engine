import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';

/**
 * Presentation component handling document file selection and upload trigger.
 */
@Component({
  selector: 'app-upload-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="upload-container">
      <h3>Ingestão de Documento para Auditoria Semântica</h3>
      <p class="subtitle">Envie um contrato, aditivo ou termo (PDF ou Imagem) para processamento no grafo RAG.</p>
      
      <div 
        class="dropzone" 
        (dragover)="onDragOver($event)" 
        (drop)="onFileDrop($event)"
        (click)="fileInput.click()">
        <input 
          #fileInput 
          type="file" 
          (change)="onFileSelected($event)" 
          accept=".pdf,.png,.jpg,.jpeg" 
          style="display: none;" />
        
        <div class="dropzone-content">
          <div class="icon">📄</div>
          <p *ngIf="!selectedFile">Arraste seu arquivo aqui ou clique para selecionar</p>
          <p *ngIf="selectedFile" class="selected-name">Arquivo selecionado: <strong>{{ selectedFile.name }}</strong></p>
          <span class="hint">Suporta contratos em PDF, PNG e JPG</span>
        </div>
      </div>

      <div class="action-bar" *ngIf="selectedFile">
        <button class="btn-submit" (click)="triggerAnalysis()">Iniciar Extração & Grafo de Equivalência</button>
      </div>
    </div>
  `,
  styles: [`
    .upload-container {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }
    h3 { font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; }
    .subtitle { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 18px; }
    .dropzone {
      border: 2px dashed var(--border-color);
      border-radius: 8px;
      padding: 36px;
      text-align: center;
      cursor: pointer;
      background: var(--bg-card);
      transition: border-color 0.2s;
    }
    .dropzone:hover { border-color: var(--accent-cyan); }
    .icon { font-size: 2.5rem; margin-bottom: 8px; }
    .hint { display: block; font-size: 0.8rem; color: var(--text-muted); margin-top: 6px; }
    .selected-name { color: var(--accent-cyan); font-weight: 500; }
    .action-bar { margin-top: 16px; text-align: right; }
    .btn-submit {
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
      border: none;
      color: #fff;
      font-weight: 600;
      padding: 10px 20px;
      border-radius: 6px;
      cursor: pointer;
    }
  `]
})
export class UploadPanelComponent {
  @Output() fileSubmitted = new EventEmitter<File>();
  selectedFile: File | null = null;

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }

  onFileDrop(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      this.selectedFile = event.dataTransfer.files[0];
    }
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
    }
  }

  triggerAnalysis() {
    if (this.selectedFile) {
      this.fileSubmitted.emit(this.selectedFile);
    }
  }
}
