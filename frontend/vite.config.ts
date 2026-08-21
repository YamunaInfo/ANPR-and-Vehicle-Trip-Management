import path from 'path';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';
import fs from 'fs';

import runtimeErrorOverlay from '@replit/vite-plugin-runtime-error-modal';

const rawPort = process.env.PORT || '5173';
const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const basePath = process.env.BASE_PATH || '/';

// onnxruntime-web WASM loader files live in node_modules dist.
// Vite 7 blocks serving public/*.mjs via JS import(). This plugin intercepts
// those requests and serves them directly as raw JS from node_modules.
const ortWasmLoaderPlugin = () => {
  const ortDistDir = path.resolve(
    import.meta.dirname,
    '..',
    'node_modules',
    '.pnpm',
    'onnxruntime-web@1.27.0',
    'node_modules',
    'onnxruntime-web',
    'dist',
  );
  const loaderPattern = /^\/ort-wasm-simd-threaded(\.[a-z]+)?\.mjs$/;

  return {
    name: 'ort-wasm-loader',
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        const url: string = req.url?.split('?')[0] ?? '';
        if (!loaderPattern.test(url)) return next();

        const filename = url.slice(1); // strip leading /
        const filePath = path.join(ortDistDir, filename);

        if (!fs.existsSync(filePath)) return next();

        res.setHeader('Content-Type', 'application/javascript');
        res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
        fs.createReadStream(filePath).pipe(res);
      });
    },
  };
};

export default defineConfig({
  base: basePath,
  plugins: [
    ortWasmLoaderPlugin(),
    react(),
    tailwindcss(),
    runtimeErrorOverlay(),
    ...(process.env.NODE_ENV !== 'production' &&
    process.env.REPL_ID !== undefined
      ? [
          await import('@replit/vite-plugin-cartographer').then((m) =>
            m.cartographer({
              root: path.resolve(import.meta.dirname, '..'),
            }),
          ),
          await import('@replit/vite-plugin-dev-banner').then((m) =>
            m.devBanner(),
          ),
        ]
      : []),
  ],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'src'),
      '@assets': path.resolve(
        import.meta.dirname,
        '..',
        'attached_assets',
      ),
      '@workspace/ai': path.resolve(import.meta.dirname, '..', 'ai', 'src', 'index.ts'),
    },
    dedupe: ['react', 'react-dom'],
  },
  root: path.resolve(import.meta.dirname),
  build: {
    outDir: path.resolve(import.meta.dirname, 'dist/public'),
    emptyOutDir: true,
  },
  // Exclude onnxruntime-web from Vite dep optimization so it doesn't
  // try to pre-bundle the WASM loader .mjs files through the module pipeline.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    port,
    strictPort: true,
    host: '0.0.0.0',
    allowedHosts: true,
    fs: {
      // Allow reading from node_modules (needed for ort-wasm files)
      strict: false,
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
    },
    headers: {
      // Required for SharedArrayBuffer used by onnxruntime-web multi-threaded WASM
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    port,
    host: '0.0.0.0',
    allowedHosts: true,
  },
});
