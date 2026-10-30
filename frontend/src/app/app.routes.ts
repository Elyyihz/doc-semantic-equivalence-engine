import { Routes } from '@angular/router';
import { DashboardComponent } from './presentation/pages/dashboard/dashboard.component';

export const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: '**', redirectTo: '' }
];
