import { type ReactNode } from 'react';
import { AlertTriangle, Check, ChevronRight, CircleHelp, Loader2, RefreshCw, Search, X } from 'lucide-react';
import { Link } from 'wouter';

export function Logo({ compact = false }: { compact?: boolean }) {
  return <Link href="/" data-testid="link-logo" className="flex items-center gap-3 text-foreground no-underline">
    <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-amber-500/50 bg-amber-500/10 text-amber-400">
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect width="18" height="18" x="3" y="3" rx="4"/><rect width="8" height="8" x="8" y="8" rx="1"/></svg>
    </div>
    {!compact && <div><span className="block text-xs font-bold tracking-wider uppercase text-foreground">ANPRX</span><span className="block text-[9px] font-semibold tracking-widest text-muted-foreground uppercase">EDGE ANPR & TRIP MANAGEMENT</span></div>}
  </Link>;
}

export function Button({ children, variant = 'primary', className = '', disabled, onClick, type = 'button', testId, title }: {
  children: ReactNode; variant?: 'primary' | 'secondary' | 'ghost' | 'danger'; className?: string; disabled?: boolean; onClick?: () => void; type?: 'button' | 'submit'; testId?: string; title?: string;
}) {
  const styles = { primary: 'bg-primary text-primary-foreground hover:brightness-110', secondary: 'bg-secondary text-secondary-foreground border border-border hover:bg-accent/10', ghost: 'text-muted-foreground hover:bg-secondary hover:text-foreground', danger: 'bg-destructive/15 text-red-200 border border-destructive/35 hover:bg-destructive/25' };
  return <button type={type} disabled={disabled} onClick={onClick} data-testid={testId} title={title} className={`inline-flex items-center justify-center gap-2 rounded-md px-3.5 py-2 text-sm font-semibold transition-all active:translate-y-px disabled:cursor-not-allowed disabled:opacity-45 ${styles[variant]} ${className}`}>{children}</button>;
}

export function Card({ children, className = '', title, action }: { children: ReactNode; className?: string; title?: string; action?: ReactNode }) {
  return <section className={`rounded-xl border border-card-border bg-card shadow-[0_12px_30px_rgba(0,0,0,.13)] ${className}`}>
    {title && <div className="flex items-center justify-between border-b border-border/70 px-5 py-4"><h2 className="text-[11px] font-bold uppercase tracking-[.16em] text-muted-foreground">{title}</h2>{action}</div>}
    {children}
  </section>;
}

export function StatusPill({ value, tone }: { value: string; tone?: 'good' | 'warn' | 'bad' | 'neutral' }) {
  const normalized = value.toLowerCase();
  const inferred = tone ?? (normalized.includes('allow') || normalized.includes('active') || normalized.includes('online') || normalized.includes('authorized') || normalized.includes('completed') || normalized.includes('inside') ? 'good' : normalized.includes('review') || normalized.includes('pending') || normalized.includes('warning') || normalized.includes('scheduled') || normalized.includes('idle') ? 'warn' : normalized.includes('deny') || normalized.includes('offline') || normalized.includes('overstay') || normalized.includes('blocked') ? 'bad' : 'neutral');
  const styles = { good: 'bg-emerald-400/10 text-emerald-300 border-emerald-400/25', warn: 'bg-amber-300/10 text-amber-200 border-amber-300/25', bad: 'bg-red-400/10 text-red-300 border-red-400/25', neutral: 'bg-slate-400/10 text-slate-300 border-slate-400/20' };
  return <span data-testid={`status-${value.replace(/\s+/g, '-').toLowerCase()}`} className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-[.12em] ${styles[inferred]}`}>{value}</span>;
}

export function Metric({ label, value, suffix, accent = 'primary', detail }: { label: string; value: string | number; suffix?: string; accent?: 'primary' | 'accent' | 'danger' | 'good'; detail?: string }) {
  const colors = { primary: 'text-primary', accent: 'text-accent', danger: 'text-red-300', good: 'text-emerald-300' };
  return <div className="rounded-xl border border-border/80 bg-card/70 p-4"><div className="mb-3 flex items-center justify-between"><span className="text-[10px] font-bold uppercase tracking-[.15em] text-muted-foreground">{label}</span><span className={`h-1.5 w-1.5 rounded-full bg-current ${colors[accent]} animate-pulse-dot`} /></div><div className={`data-text text-2xl font-bold ${colors[accent]}`}>{value}<span className="ml-1 text-xs font-normal text-muted-foreground">{suffix}</span></div>{detail && <p className="mt-1 text-[11px] text-muted-foreground">{detail}</p>}</div>;
}

export function PageHeader({ eyebrow, title, description, children }: { eyebrow: string; title: string; description?: string; children?: ReactNode }) {
  return <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between"><div><p className="mb-2 text-[10px] font-bold uppercase tracking-[.22em] text-primary">{eyebrow}</p><h1 className="text-2xl font-bold tracking-tight text-foreground md:text-3xl">{title}</h1>{description && <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p>}</div>{children && <div className="flex flex-wrap gap-2">{children}</div>}</div>;
}

export function LoadingRows({ count = 5 }: { count?: number }) {
  return <div className="space-y-3 p-5" data-testid="loading-state">{Array.from({ length: count }).map((_, i) => <div key={i} className="h-10 animate-pulse rounded-md bg-secondary/70" />)}</div>;
}

export function ErrorState({ message = 'Signal unavailable', retry }: { message?: string; retry?: () => void }) {
  return <div className="flex flex-col items-center justify-center gap-3 p-10 text-center" data-testid="error-state"><AlertTriangle className="h-6 w-6 text-amber-300" /><p className="text-sm text-muted-foreground">{message}</p>{retry && <Button variant="secondary" onClick={retry} testId="button-retry"><RefreshCw className="h-3.5 w-3.5" />Retry</Button>}</div>;
}

export function EmptyState({ title, detail, action }: { title: string; detail: string; action?: ReactNode }) {
  return <div className="flex flex-col items-center justify-center p-12 text-center" data-testid="empty-state"><CircleHelp className="mb-3 h-7 w-7 text-muted-foreground/70" /><h3 className="text-sm font-semibold">{title}</h3><p className="mt-1 max-w-sm text-xs text-muted-foreground">{detail}</p>{action && <div className="mt-5">{action}</div>}</div>;
}

export function DataTable({ children, headers }: { children: ReactNode; headers: string[] }) {
  return <div className="overflow-x-auto scrollbar-thin"><table className="w-full min-w-[720px] text-left"><thead><tr className="border-b border-border/70">{headers.map((h) => <th key={h} className="px-5 py-3 text-[10px] font-bold uppercase tracking-[.15em] text-muted-foreground">{h}</th>)}</tr></thead>{children}</table></div>;
}

export function SearchBox({ value, onChange, placeholder = 'Search records' }: { value: string; onChange: (value: string) => void; placeholder?: string }) {
  return <label className="relative block min-w-[220px] flex-1"><Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" /><input data-testid="input-search" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="h-10 w-full rounded-md border border-border bg-secondary/60 pl-9 pr-3 text-sm outline-none transition focus:border-primary/60 focus:ring-2 focus:ring-primary/15" /></label>;
}

export function IconButton({ children, label, onClick, testId }: { children: ReactNode; label: string; onClick?: () => void; testId?: string }) {
  return <button aria-label={label} title={label} data-testid={testId} onClick={onClick} className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition hover:bg-secondary hover:text-foreground">{children}</button>;
}

export function Field({ label, value, onChange, placeholder, type = 'text', required = false }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string; type?: string; required?: boolean }) {
  return <label className="block"><span className="mb-1.5 block text-[10px] font-bold uppercase tracking-[.13em] text-muted-foreground">{label}{required && <span className="text-primary"> *</span>}</span><input required={required} type={type} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="h-10 w-full rounded-md border border-border bg-background/70 px-3 text-sm outline-none transition placeholder:text-muted-foreground/50 focus:border-primary/60 focus:ring-2 focus:ring-primary/15" /></label>;
}

export function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: string[] }) {
  return <label className="block"><span className="mb-1.5 block text-[10px] font-bold uppercase tracking-[.13em] text-muted-foreground">{label}</span><select value={value} onChange={(e) => onChange(e.target.value)} className="h-10 w-full rounded-md border border-border bg-background/70 px-3 text-sm outline-none focus:border-primary/60">{options.map(o => <option key={o} value={o}>{o}</option>)}</select></label>;
}

export function Modal({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm" role="dialog" aria-modal="true"><div className="w-full max-w-lg rounded-xl border border-border bg-card shadow-2xl"><div className="flex items-center justify-between border-b border-border px-5 py-4"><h2 className="font-semibold">{title}</h2><IconButton label="Close dialog" onClick={onClose} testId="button-close-dialog"><X className="h-4 w-4" /></IconButton></div><div className="p-5">{children}</div></div></div>;
}

export function TinyBar({ value, max, color = 'bg-primary' }: { value: number; max: number; color?: string }) {
  return <div className="h-1.5 overflow-hidden rounded-full bg-secondary"><div className={`h-full rounded-full ${color}`} style={{ width: `${Math.min(100, max ? value / max * 100 : 0)}%` }} /></div>;
}

export function Notice({ children, kind = 'good' }: { children: ReactNode; kind?: 'good' | 'bad' }) {
  return <div className={`mb-4 flex items-center gap-2 rounded-md border px-3 py-2 text-xs ${kind === 'good' ? 'border-emerald-400/25 bg-emerald-400/10 text-emerald-200' : 'border-red-400/25 bg-red-400/10 text-red-200'}`} data-testid="status-notice">{kind === 'good' ? <Check className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}{children}</div>;
}

export function Busy({ label = 'Working' }: { label?: string }) { return <span className="inline-flex items-center gap-2"><Loader2 className="h-3.5 w-3.5 animate-spin" />{label}</span>; }

export function DetailLink({ href, children }: { href: string; children: ReactNode }) { return <Link href={href} className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline" data-testid={`link-${href.replace(/\//g, '')}`}>{children}<ChevronRight className="h-3 w-3" /></Link>; }