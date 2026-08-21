import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { ShieldCheck, Eye, EyeOff, Loader2, ArrowRight, CheckCircle2 } from 'lucide-react';
import { useAuth } from '@/lib/auth';

interface LoginPageProps {
  initialMode?: 'signin' | 'signup';
}

export default function LoginPage({ initialMode = 'signup' }: LoginPageProps) {
  const [location, setLocation] = useLocation();
  const { login, signup, isLoading } = useAuth();

  const [mode, setMode] = useState<'signin' | 'signup'>(() => {
    if (location === '/login') return 'signin';
    if (location === '/signup') return 'signup';
    return initialMode;
  });
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!email.trim()) {
      setError('Please enter your email address.');
      return;
    }

    if (mode === 'signup' && !fullName.trim()) {
      setError('Please enter your full name.');
      return;
    }

    if (!password) {
      setError('Please enter a password.');
      return;
    }

    try {
      if (mode === 'signup') {
        await signup(fullName, email, password, 'guard');
        setSuccess('Account created successfully! Please enter your password to sign in.');
        setPassword('');
        setMode('signin');
      } else {
        await login(email, password);
        setSuccess('Signed in successfully! Redirecting to dashboard...');
        setTimeout(() => {
          setLocation('/');
        }, 400);
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please try again.');
    }
  };

  return (
    <div className="relative min-h-[100dvh] w-full bg-[#070A0F] text-white flex items-center justify-center p-4 md:p-8 overflow-hidden font-sans">
      {/* Background Subtle Grid Pattern */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-40"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.04) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
        }}
      />

      {/* Ambient Lighting Gradients */}
      <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-amber-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-96 h-96 bg-blue-500/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Main Split Authentication Card */}
      <div className="relative z-10 w-full max-w-[1000px] rounded-3xl border border-[#1E2638] bg-[#0E141E]/95 shadow-[0_24px_80px_rgba(0,0,0,0.85)] backdrop-blur-xl overflow-hidden grid grid-cols-1 lg:grid-cols-2">
        
        {/* Left Side: Branding, Visual Graphics & Hero */}
        <div className="relative bg-[#0B1019] p-8 md:p-12 flex flex-col justify-between border-b lg:border-b-0 lg:border-r border-[#1E2638] overflow-hidden">
          {/* Subtle Top-Left Ambient Gold Light */}
          <div className="absolute -left-16 -top-16 w-64 h-64 bg-amber-500/15 rounded-full blur-3xl pointer-events-none" />

          {/* Decorative Concentric Rings Watermark in Top-Right */}
          <div className="absolute -right-6 top-4 pointer-events-none opacity-60">
            <svg width="180" height="180" viewBox="0 0 160 160" fill="none">
              <circle cx="90" cy="70" r="64" stroke="#F59E0B" strokeOpacity="0.22" strokeWidth="1.2" strokeDasharray="4 2" />
              <circle cx="90" cy="70" r="44" stroke="#F59E0B" strokeOpacity="0.3" strokeWidth="1.4" />
              <circle cx="90" cy="70" r="24" stroke="#F59E0B" strokeOpacity="0.4" strokeWidth="1.2" />
            </svg>
          </div>

          {/* Top Logo */}
          <div className="relative z-10 flex items-center gap-3">
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-400 shadow-inner">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect width="18" height="18" x="3" y="3" rx="4" />
                <rect width="8" height="8" x="8" y="8" rx="1" />
              </svg>
            </div>
            <div>
              <span className="block text-xs font-bold tracking-[0.14em] uppercase text-white">EDGE ANPR</span>
              <span className="block text-[9px] font-semibold tracking-[0.2em] text-gray-400 uppercase">TRIP MANAGEMENT</span>
            </div>
          </div>

          {/* Bottom Hero Content */}
          <div className="relative z-10 pt-16 lg:pt-24 space-y-4">
            <div className="inline-flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.18em] text-[#F59E0B]">
              <span>EDGE ANPR & TRIP MANAGEMENT</span>
            </div>

            <h1 className="text-3xl md:text-[38px] font-bold leading-[1.18] tracking-tight text-white">
              Secure access for every vehicle movement.
            </h1>

            <p className="text-sm text-gray-400 leading-relaxed max-w-md">
              Monitor vehicle access with AI-powered ANPR and trip management.
            </p>

            {/* Status Pills */}
            <div className="flex flex-wrap items-center gap-4 pt-4 text-xs font-bold tracking-wider">
              <div className="flex items-center gap-2 text-emerald-400 uppercase text-[10px] tracking-[0.14em]">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                LIVE MONITORING
              </div>
              <div className="flex items-center gap-1.5 text-amber-400/90 uppercase text-[10px] tracking-[0.14em]">
                <ShieldCheck className="h-3.5 w-3.5" />
                SECURE ACCESS
              </div>
            </div>
          </div>
        </div>

        {/* Right Side: Authentication Form */}
        <div className="p-8 md:p-12 bg-[#0E141E] flex flex-col justify-between">
          <div>
            {/* Top Switch Pills (Sign In / Sign Up) */}
            <div className="flex justify-end mb-8">
              <div className="inline-flex items-center bg-[#131B28] p-1 rounded-full border border-[#1E2638]">
                <button
                  type="button"
                  onClick={() => {
                    setMode('signin');
                    setError(null);
                    setSuccess(null);
                  }}
                  className={`px-5 py-1.5 text-xs font-semibold rounded-full transition-all duration-200 ${
                    mode === 'signin'
                      ? 'bg-[#F59E0B] text-black shadow-md font-bold'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMode('signup');
                    setError(null);
                    setSuccess(null);
                  }}
                  className={`px-5 py-1.5 text-xs font-semibold rounded-full transition-all duration-200 ${
                    mode === 'signup'
                      ? 'bg-[#F59E0B] text-black shadow-md font-bold'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  Sign Up
                </button>
              </div>
            </div>

            {/* Eyebrow & Title */}
            <div className="mb-6 space-y-1">
              <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-[#F59E0B]">
                ACCESS PORTAL
              </p>
              <h2 className="text-2xl font-bold tracking-tight text-white">
                {mode === 'signin' ? 'Welcome back' : 'Create your account'}
              </h2>
            </div>

            {/* Error or Success feedback */}
            {error && (
              <div className="mb-5 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-xs text-red-300 animate-slide-in">
                {error}
              </div>
            )}
            {success && (
              <div className="mb-5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-2.5 text-xs text-emerald-300 flex items-center gap-2 animate-slide-in">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                <span>{success}</span>
              </div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Full Name field (Sign Up only) */}
              {mode === 'signup' && (
                <div className="space-y-1.5 animate-slide-in">
                  <label className="block text-[10px] font-bold uppercase tracking-[0.14em] text-gray-300">
                    FULL NAME <span className="text-[#F59E0B]">*</span>
                  </label>
                  <input
                    type="text"
                    required
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Your name"
                    className="w-full rounded-lg border border-[#1E2638] bg-[#131B28] px-3.5 py-2.5 text-sm text-white placeholder-gray-500 outline-none transition focus:border-[#F59E0B] focus:ring-1 focus:ring-[#F59E0B]"
                  />
                </div>
              )}

              {/* Email Address */}
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold uppercase tracking-[0.14em] text-gray-300">
                  EMAIL ADDRESS <span className="text-[#F59E0B]">*</span>
                </label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@example.com"
                  className="w-full rounded-lg border border-[#1E2638] bg-[#131B28] px-3.5 py-2.5 text-sm text-white placeholder-gray-500 outline-none transition focus:border-[#F59E0B] focus:ring-1 focus:ring-[#F59E0B]"
                />
              </div>

              {/* Password */}
              <div className="space-y-1.5">
                <label className="block text-[10px] font-bold uppercase tracking-[0.14em] text-gray-300">
                  PASSWORD <span className="text-[#F59E0B]">*</span>
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={mode === 'signup' ? 'Create a password' : '••••••••'}
                    className="w-full rounded-lg border border-[#1E2638] bg-[#131B28] px-3.5 py-2.5 pr-10 text-sm text-white placeholder-gray-500 outline-none transition focus:border-[#F59E0B] focus:ring-1 focus:ring-[#F59E0B]"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-200"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full mt-6 flex items-center justify-center gap-2 rounded-lg bg-[#F59E0B] py-3 text-sm font-bold text-black transition-all hover:bg-[#E59306] active:translate-y-px disabled:opacity-50 shadow-lg shadow-amber-500/10 cursor-pointer"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : mode === 'signin' ? (
                  <>
                    <span>Sign in</span>
                    <ArrowRight className="h-4 w-4" />
                  </>
                ) : (
                  <>
                    <span>Create account</span>
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Bottom Switch Link */}
          <div className="mt-6 text-center text-xs text-gray-400">
            {mode === 'signin' ? (
              <p>
                Need an account?{' '}
                <button
                  type="button"
                  onClick={() => {
                    setMode('signup');
                    setError(null);
                    setSuccess(null);
                  }}
                  className="font-bold text-[#F59E0B] hover:underline"
                >
                  Create one
                </button>
              </p>
            ) : (
              <p>
                Already have an account?{' '}
                <button
                  type="button"
                  onClick={() => {
                    setMode('signin');
                    setError(null);
                    setSuccess(null);
                  }}
                  className="font-bold text-[#F59E0B] hover:underline"
                >
                  Sign in
                </button>
              </p>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
